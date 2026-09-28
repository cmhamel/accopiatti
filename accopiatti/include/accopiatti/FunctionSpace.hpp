#pragma once
#include <accopiatti/DofManager.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include <Intrepid2_HGRAD_HEX_C1_FEM.hpp>
#include <Intrepid2_Orientation.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>

namespace accopiatti {

template<Scalar T>
struct QuadratureWorkset {
    int num_qp;
    QuadraturePoints<T> q_pts;
    QuadratureWeights<T> q_wts;

    QuadratureWorkset() = default;

    explicit QuadratureWorkset(
        const Teuchos::RCP<Basis<T>>& basis,
        const int& quadrature_order
    ) {
        const shards::CellTopology cell_topo = basis->getBaseCellTopology();
        auto dim = cell_topo.getDimension();
        auto quadrature = Intrepid2::DefaultCubatureFactory::create<Device, T, T>(cell_topo, quadrature_order);
        num_qp = static_cast<int>(quadrature->getNumPoints());
        std::cout << "Num qpoints = " << num_qp;
        q_pts = QuadraturePoints<T>("quadrature_points", num_qp, dim);
        q_wts = QuadratureWeights<T>("quadrature_weights", num_qp);
        quadrature->getCubature(q_pts, q_wts);
    }
};

template<Scalar T, int Dim>
struct GeometryBasisWorkset {
    using CellTools = Intrepid2::CellTools<Device>;
    using FSpaceTools = Intrepid2::FunctionSpaceTools<Device>;

    int num_basis;
    int num_elements;
    int num_qp;
    Teuchos::RCP<Basis<T>> basis;
    CoordinatesView<T, Dim> el_coords;
    // Kokkos::DynRankView<T, Device> el_coords;
    Jacobians<T, Dim> jacs;
    JacobianDets<T> jac_dets;
    JacobianInverses<T> jac_invs;
    JxWs<T> jxws;
    ReferenceBasisGradients<T> grad_N_xis;
    ReferenceBasisValues<T> Ns;

    GeometryBasisWorkset() = default;

    explicit GeometryBasisWorkset(
        const Teuchos::RCP<Basis<T>>& geometry_basis_,
        const QuadratureWorkset<T>& quad_ws,
        CoordinatesView<T, Dim>& el_coords_
        // Kokkos::DynRankView<T, Device>& el_coords_
    ) {
        std::cout << "El coords size = (" << el_coords_.extent(0) << ", " << el_coords_.extent(1) << std::endl;
        basis = geometry_basis_;
        el_coords = el_coords_;
        num_elements = el_coords.extent(0);
        num_qp = quad_ws.num_qp;
        num_basis = static_cast<int>(geometry_basis_->getCardinality());

        Ns = ReferenceBasisValues<T>("reference_basis_values", num_basis, num_qp);
        grad_N_xis = ReferenceBasisGradients<T>("reference_basis_gradients", num_basis, num_qp, Dim);

        // setup basis values
        basis->getValues(Ns, quad_ws.q_pts, Intrepid2::OPERATOR_VALUE);
        basis->getValues(grad_N_xis, quad_ws.q_pts, Intrepid2::OPERATOR_GRAD);

        // setup jacobian scratch
        jacs = Jacobians<T, Dim>("jacobians", num_elements, num_qp);
        jac_invs = JacobianInverses<T>("jacobian_inverses", num_elements, num_qp, Dim, Dim);
        jac_dets = JacobianDets<T>("jacobian_dets", num_elements, num_qp);
        jxws = JxWs<T>("JxWs", num_elements, num_qp);

        update_jacobians(quad_ws);
    }

    void debug_dump_bad_elements(T tol) const {
        auto h_coords  = Kokkos::create_mirror_view(el_coords);
        auto h_jacdets = Kokkos::create_mirror_view(jac_dets);
        Kokkos::deep_copy(h_coords, el_coords);
        Kokkos::deep_copy(h_jacdets, jac_dets);

        for (int c = 0; c < num_elements; ++c) {
            T min_det =  std::numeric_limits<T>::max();
            T max_det = -std::numeric_limits<T>::max();
            for (int p = 0; p < num_qp; ++p) {
                min_det = std::min(min_det, h_jacdets(c, p));
                max_det = std::max(max_det, h_jacdets(c, p));
            }

            const bool sign_flip = (min_det < 0.0 && max_det > 0.0);
            const bool near_zero = (std::abs(max_det) < tol && std::abs(min_det) < tol);

            if (sign_flip || near_zero) {
                std::cout << "== Suspicious element (workset-local index " << c
                        << "), det range [" << min_det << ", " << max_det << "] ==\n";

                // Raw nodal coordinates, exactly as fed to CellTools::setJacobian.
                for (int n = 0; n < num_basis; ++n) {
                    std::cout << "  node " << n << ": (";
                    for (int d = 0; d < Dim; ++d) {
                        std::cout << h_coords(c, n, d) << (d + 1 < Dim ? ", " : "");
                    }
                    std::cout << ")\n";
                }

                // Flag exact or near-duplicate nodes -- the single most common
                // cause of a "folded" / zero-volume element.
                for (int n1 = 0; n1 < num_basis; ++n1) {
                    for (int n2 = n1 + 1; n2 < num_basis; ++n2) {
                        T dist2 = 0.0;
                        for (int d = 0; d < Dim; ++d) {
                            const T diff = h_coords(c, n1, d) - h_coords(c, n2, d);
                            dist2 += diff * diff;
                        }
                        if (dist2 < tol * tol) {
                            std::cout << "  !! nodes " << n1 << " and " << n2
                                    << " are coincident (dist = " << std::sqrt(dist2) << ")\n";
                        }
                    }
                }
            }
        } 
    }

    double get_volume() const {
        double vol = 0.0;
        auto& jxws_ = jxws;
        Kokkos::parallel_reduce(
            "accopiatti::get_volume",
            Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>>({0, 0}, {num_elements, num_qp}),
            KOKKOS_LAMBDA(const int c, const int p, double& update) {
                update += jxws_(c, p);
            },
            vol
        );
        return vol;
    }

    void update_jacobians(const QuadratureWorkset<T>& quad_ws) {
        CellTools::setJacobian(jacs, quad_ws.q_pts, el_coords, basis);
        CellTools::setJacobianInv(jac_invs, jacs);
        CellTools::setJacobianDet(jac_dets, jacs);
        FSpaceTools::computeCellMeasure(jxws, jac_dets, quad_ws.q_wts);
    }
};

// TODO only works for total lagrange type implementations currently
template<Scalar T, int Dim>
struct SolutionBasisWorkset {
    using CellTools = Intrepid2::CellTools<Device>;
    using FSpaceTools = Intrepid2::FunctionSpaceTools<Device>;

    int num_basis;
    int num_elements;
    int num_qp;
    Teuchos::RCP<Basis<T>> basis;
    Teuchos::RCP<Basis<T>> geometry_basis;
    PhysicalGradients<T> physical_grads;
    PhysicalPoints<T, Dim> physical_q_pts;
    ReferenceBasisGradients<T> grad_N_xis;
    ReferenceBasisValues<T> Ns;

    SolutionBasisWorkset() = default;
    SolutionBasisWorkset(
        const Teuchos::RCP<Basis<T>>& basis_,
        const QuadratureWorkset<T>& quad_workset_,
        const GeometryBasisWorkset<T, Dim>& geom_workset_
    ) {
        basis = basis_;
        num_basis = static_cast<int>(basis_->getCardinality());
        num_elements = geom_workset_.num_elements;
        num_qp = quad_workset_.num_qp;
        const int num_geom_basis = geom_workset_.num_basis;

        Ns = ReferenceBasisValues<T>("solution_basis_values", num_basis, num_qp);
        grad_N_xis = ReferenceBasisGradients<T>("solution_basis_gradients", num_basis, num_qp, Dim);

        // setup basis values
        basis->getValues(Ns, quad_workset_.q_pts, Intrepid2::OPERATOR_VALUE);
        basis->getValues(grad_N_xis, quad_workset_.q_pts, Intrepid2::OPERATOR_GRAD);

        // setup jacobian scratch
        const int num_elements = geom_workset_.el_coords.extent(0);

        // physical points and gradients
        physical_q_pts = PhysicalPoints<T, Dim>("physical_quadrature_pts", num_elements, num_qp);
        physical_grads = PhysicalGradients<T>("physical_gradients", num_elements, num_basis, num_qp, Dim);

        // update_jacobians();
        update_physical_quadrature_points(geom_workset_);
        update_physical_gradients(geom_workset_);
    }

    void update_physical_gradients(const GeometryBasisWorkset<T, Dim>& geom_workset) {
        FSpaceTools::HGRADtransformGRAD(physical_grads, geom_workset.jac_invs, grad_N_xis);
    }

    void update_physical_quadrature_points(const GeometryBasisWorkset<T, Dim>& geom_basis) {
        auto physical_q_pts_ = physical_q_pts;
        Kokkos::parallel_for("accopiatti::physical_quadrature_points",
            Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>>({0, 0}, {num_elements, num_qp}),
            KOKKOS_LAMBDA(const int c, const int p) {
                T x[Dim] = {};
                for (int n = 0; n < geom_basis.num_basis; ++n) {
                    const T Nn = geom_basis.Ns(n, p);
                    for (int d = 0; d < Dim; ++d) {
                        x[d] += Nn * geom_basis.el_coords(c, n, d);
                    }
                }
                for (int d = 0; d < Dim; ++d) {
                    physical_q_pts_(c, p, d) = x[d];
                }
            }
        );
    }
};

template<Scalar T, int Dim, int NumSol>
struct BlockWorkset {
    std::string block_name;
    GeometryBasisWorkset<T, Dim> geom_workset;
    Kokkos::View<std::size_t*, Device> local_el_ids;
    QuadratureWorkset<T> quad_workset;
    std::array<SolutionBasisWorkset<T, Dim>, NumSol> sol_worksets;
    Kokkos::View<GlobalOrdinal**, Device> global_dof_ids;

    BlockWorkset() = default;
    explicit BlockWorkset(
        const std::string& block_name_,
        const GeometryBasisWorkset<T, Dim>& geom_workset_,
        const Kokkos::View<std::size_t*, Device>& local_el_ids_,
        const QuadratureWorkset<T>& quad_workset_,
        const std::array<SolutionBasisWorkset<T, Dim>, NumSol>& sol_worksets_,
        const Kokkos::View<GlobalOrdinal**, Device>& global_dof_ids_
    ) 
        :
        block_name(block_name_),
        geom_workset(geom_workset_),
        local_el_ids(local_el_ids_),
        quad_workset(quad_workset_),
        sol_worksets(sol_worksets_),
        global_dof_ids(global_dof_ids_) {}

    // TODO need to eventually plumb new coordinates so we can do things like
    // updated lagrange methods
    void update_geometry() {
        geom_workset.update_jacobians(quad_workset);
        for (int i = 0; i < NumSol; ++i) {
            sol_worksets[i].update_physical_points(geom_workset);
            sol_worksets[i].update_physical_gradients(geom_workset);
        }
    }
};

template<Scalar T, int Dim, int NumSol>
class FunctionSpace {
public:
    explicit FunctionSpace(
        const Mesh& mesh,
        DofManager<T, Dim>& dof_manager,
        const int& workset_size
    ) {
        auto bulk = mesh.get_bulk_data();
        auto block_names = mesh.get_element_block_names();
        auto* coord_field = mesh.get_coordinate_field<T>();
        auto field_names = dof_manager.get_solution_field_names();
        for (auto& block_name : block_names) {
            auto block_el_coords = dof_manager.get_dof_coordinates(block_name);
            const auto& geometry_basis = dof_manager.geom_basis_map[block_name];
            const int& num_geom_basis = geometry_basis->getCardinality();
            auto num_block_dofs = dof_manager.dof_manager->getElementBlockGIDCount(block_name);
            auto buckets = mesh.get_local_elements(block_name);
            std::cout << "In block = " << block_name << std::endl;
            for (const stk::mesh::Bucket* bucket : buckets) {
                const stk::topology topology = bucket->topology();
                const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
                const int bucket_size = static_cast<int>(bucket->size());
                std::map<int, int> stk_to_interpid2_dof_map = dof_manager.stk_to_intrepid2_dof_maps[block_name];
                for (int bucket_offset = 0; bucket_offset < bucket_size; bucket_offset += workset_size) {
                    const int num_els_in_workset = std::min(workset_size, bucket_size - bucket_offset);
                    auto el_coords = CoordinatesView<T, Dim>("el_coords_" + block_name, num_els_in_workset, num_nodes_per_el);
                    auto el_coords_2 = CoordinatesView<T, Dim>("el_coords_" + block_name, num_els_in_workset, num_nodes_per_el);
                    auto h_el_nodes = Kokkos::create_mirror_view(el_coords);
                    auto local_elem_ids = Kokkos::View<std::size_t*, Device>("local_elem_ids_" + block_name, num_els_in_workset);

                    auto h_local_elem_ids = Kokkos::create_mirror_view(local_elem_ids);
                    auto block_global_dof_ids = Kokkos::View<GlobalOrdinal**, Device>(block_name + "_global_dof_ids", num_els_in_workset, num_block_dofs);

                    for (int c = 0; c < num_els_in_workset; ++c) {
                        const stk::mesh::Entity elem = (*bucket)[bucket_offset + c];
                        auto elem_id = mesh.mesh->elementLocalId(elem);

                        h_local_elem_ids(c) = static_cast<std::size_t>(elem_id);
                        const unsigned elem_num_nodes = bulk->num_nodes(elem);
                        const stk::mesh::Entity* elem_nodes = bulk->begin_nodes(elem);

                        // element coordinates, should be a kernel in the future
                        for (int n = 0; n < num_nodes_per_el; ++n) {
                        // for (int n = 0; n < geometry_basis->getCardinality(); ++n) {
                            const stk::mesh::Entity node = elem_nodes[n];
                            const T* x = stk::mesh::field_data(*coord_field, node);
                            const int n_temp = stk_to_interpid2_dof_map[n];
                            // error checking here on x not being nullptr
                            for (int d = 0; d < Dim; ++d) {
                                // // h_el_nodes(c, n_temp, d) = x[d];
                                h_el_nodes(c, n, d) = x[d];

                                el_coords_2(c, n, d) = block_el_coords(elem_id, n, d);
                            }
                        }
                        // global ordinals
                        std::vector<GlobalOrdinal> gids;
                        dof_manager.get_element_global_dof_ids(block_name, elem_id, gids);
                        for (int i = 0; i < gids.size(); ++i) {
                            block_global_dof_ids(c, i) = gids[i];
                        }
                    }
                    Kokkos::deep_copy(el_coords, h_el_nodes);

                    // auto el_coords_2 = dof_manager.get_dof_coordinates(block_name, local_elem_ids_2);

                    // const int quadrature_order = static_cast<int>(geometry_basis->getCardinality());
                    // TODO read from input file.
                    const int quadrature_order = 2;
                    auto quad_ws = QuadratureWorkset<T>(geometry_basis, quadrature_order);
                    auto geom_ws = GeometryBasisWorkset<T, Dim>(geometry_basis, quad_ws, el_coords_2);

                    // now we can loop over different field types
                    std::array<SolutionBasisWorkset<T, Dim>, NumSol> sol_worksets;
                    for (int sol_id = 0; sol_id < field_names.size(); ++sol_id) {
                        auto sol_basis = dof_manager.sol_basis_map[block_name][field_names[sol_id]];
                        auto sol_ws = SolutionBasisWorkset<T, Dim>(sol_basis, quad_ws, geom_ws);
                        sol_worksets[sol_id] = sol_ws;
                    }
                    auto block_workset = BlockWorkset<T, Dim, NumSol>(
                        block_name,
                        geom_ws,
                        local_elem_ids,
                        quad_ws, sol_worksets, block_global_dof_ids
                    );
                    block_worksets.push_back(block_workset);
                }
            }
        }
    }

    auto& get_block_worksets() {
        return block_worksets;
    }

// private:
    std::vector<BlockWorkset<T, Dim, NumSol>> block_worksets;
};

} // end namespace accopiatti
