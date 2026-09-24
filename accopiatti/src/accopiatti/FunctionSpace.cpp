#include <accopiatti/FunctionSpace.hpp>

namespace accopiatti {

template<Scalar T, int Dim>
ElementBasisWorkset<T, Dim>::ElementBasisWorkset(
    const Teuchos::RCP<Basis<T>>& basis_,
    const Teuchos::RCP<Basis<T>>& geometry_basis_,
    const int integration_order,
    CoordinatesView<T, Dim>& el_coords_
) {
    basis = basis_;
    el_coords = el_coords_;
    geometry_basis = geometry_basis_;
    num_basis = static_cast<int>(basis->getCardinality());
    num_geom_basis = static_cast<int>(geometry_basis->getCardinality());

    const shards::CellTopology cell_topo = basis->getBaseCellTopology();
    auto quadrature = Intrepid2::DefaultCubatureFactory::create<Device, T, T>(cell_topo, integration_order);
    num_qp = static_cast<int>(quadrature->getNumPoints());
    q_pts = QuadraturePoints<T>("quadrature_points", num_qp, Dim);
    q_wts = QuadratureWeights<T>("quadrature_weights", num_qp);

    Ns = ReferenceBasisValues<T>("reference_basis_values", num_basis, num_qp);
    geom_Ns = ReferenceBasisValues<T>("reference_geometry_basis_values", num_geom_basis, num_qp);
    grad_N_xis = ReferenceBasisGradients<T>("reference_basis_gradients", num_basis, num_qp, Dim);
    geom_grad_N_xis = ReferenceBasisGradients<T>("reference_geometry_basis_gradients", num_geom_basis, num_qp, Dim);

    // setup quadrature
    quadrature->getCubature(q_pts, q_wts);

    // setup basis values
    basis->getValues(Ns, q_pts, Intrepid2::OPERATOR_VALUE);
    basis->getValues(grad_N_xis, q_pts, Intrepid2::OPERATOR_GRAD);
    geometry_basis->getValues(geom_Ns, q_pts, Intrepid2::OPERATOR_VALUE);
    geometry_basis->getValues(geom_grad_N_xis, q_pts, Intrepid2::OPERATOR_GRAD);

    // setup jacobian scratch
    num_elements = el_coords.extent(0);
    jacs = Jacobians<T, Dim>("jacobians", num_elements, num_qp);
    jac_invs = JacobianInverses<T>("jacobian_inverses", num_elements, num_qp, Dim, Dim);
    jac_dets = JacobianDets<T>("jacobian_dets", num_elements, num_qp);
    jxws = JxWs<T>("JxWs", num_elements, num_qp);

    // physical points and gradients
    physical_q_pts = PhysicalPoints<T, Dim>("physical_quadrature_pts", num_elements, num_qp);
    physical_grads = PhysicalGradients<T>("physical_gradients", num_elements, num_basis, num_qp, Dim);

    update_jacobians();
    update_physical_quadrature_points();
    update_physical_gradients();

    double vol = 0.0;
    for (int i = 0; i < jac_dets.extent(0); ++i) {
        for (int j = 0; j < jac_dets.extent(1); ++j) {
            // std::cout << "jxws(" << i << ", " << j << ") = " << jac_dets(i, j)  << std::endl;
            vol += jac_dets(i, j);
        }
    }
    std::cout << "Volume = " << vol << std::endl;

    // debug_dump_bad_elements();
}

template<Scalar T, int Dim>
void ElementBasisWorkset<T, Dim>::debug_dump_bad_elements(T tol) const {
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
            for (int n = 0; n < num_geom_basis; ++n) {
                std::cout << "  node " << n << ": (";
                for (int d = 0; d < Dim; ++d) {
                    std::cout << h_coords(c, n, d) << (d + 1 < Dim ? ", " : "");
                }
                std::cout << ")\n";
            }

            // Flag exact or near-duplicate nodes -- the single most common
            // cause of a "folded" / zero-volume element.
            for (int n1 = 0; n1 < num_geom_basis; ++n1) {
                for (int n2 = n1 + 1; n2 < num_geom_basis; ++n2) {
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

template<Scalar T, int Dim>
void ElementBasisWorkset<T, Dim>::update_jacobians() {
    CellTools::setJacobian(jacs, q_pts, el_coords, geometry_basis);
    CellTools::setJacobianInv(jac_invs, jacs);
    CellTools::setJacobianDet(jac_dets, jacs);
    FSpaceTools::computeCellMeasure(jxws, jac_dets, q_wts);
}

// TODO hardcoded for Hgrad right now
template<Scalar T, int Dim>
void ElementBasisWorkset<T, Dim>::update_physical_gradients() {
    FSpaceTools::HGRADtransformGRAD(physical_grads, jac_invs, grad_N_xis);
}

template<Scalar T, int Dim>
void ElementBasisWorkset<T, Dim>::update_physical_quadrature_points() {
    Kokkos::parallel_for("accopiatti::physical_quadrature_points",
        Kokkos::MDRangePolicy<Device, Kokkos::Rank<2>>({0, 0}, {num_elements, num_qp}),
        KOKKOS_LAMBDA(const int c, const int p) {
            T x[Dim] = {};
            for (int n = 0; n < num_geom_basis; ++n) {
                const T Nn = geom_Ns(n, p);
                for (int d = 0; d < Dim; ++d) {
                    x[d] += Nn * el_coords(c, n, d);
                }
            }
            for (int d = 0; d < Dim; ++d) {
                physical_q_pts(c, p, d) = x[d];
            }
        }
    );
}

template class ElementBasisWorkset<double, 1>;
template class ElementBasisWorkset<double, 2>;
template class ElementBasisWorkset<double, 3>;

template<Scalar T, int Dim>
FunctionSpace<T, Dim>::FunctionSpace(
    const Mesh& mesh,
    const std::string& block_name,
    const FunctionSpaceHelper& fspace_helper,
    const int workset_size
) {
    auto bulk = mesh.mesh->getBulkData();
    auto meta = mesh.mesh->getMetaData();
    const int space_dim = static_cast<int>(meta->spatial_dimension());
    stk::mesh::Part* block_part = meta->get_part(block_name);
    stk::topology topology = block_part->topology();
    auto basis = BasisFactory<T>::create(topology, fspace_helper.function_space, fspace_helper.basis_order);
    auto geometry_basis = BasisFactory<T>::create_geometry(topology);

    const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
    if (num_nodes_per_el != static_cast<int>(geometry_basis->getCardinality())) {
        throw std::runtime_error("FunctionSpace: STK node count != geometry basis cardinality");
    }

    // setup coordinates
    auto* coord_field = meta->get_field<T>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::Selector selector = 
        stk::mesh::Selector(*block_part) &
        stk::mesh::Selector(meta->locally_owned_part());
    const stk::mesh::BucketVector& buckets = bulk->get_buckets(stk::topology::ELEM_RANK, selector);
    std::vector<ElementBasisWorkset<T, Dim>> worksets;
    
    for (const stk::mesh::Bucket* bucket : buckets) {
        const stk::topology topology = bucket->topology();
        const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
        const int bucket_size = static_cast<int>(bucket->size());
        for (int bucket_offset = 0; bucket_offset < bucket_size; bucket_offset += workset_size) {
            const int num_els_in_workset = std::min(workset_size, bucket_size - bucket_offset);
            auto el_coords = CoordinatesView<T, Dim>("el_coords_" + block_name, num_els_in_workset, num_nodes_per_el);
            auto h_el_nodes = Kokkos::create_mirror_view(el_coords);
            for (int c = 0; c < num_els_in_workset; ++c) {
                const stk::mesh::Entity elem = (*bucket)[bucket_offset + c];
                const unsigned elem_num_nodes = bulk->num_nodes(elem);
                // error check this
                const stk::mesh::Entity* elem_nodes = bulk->begin_nodes(elem);

                for (int n = 0; n < num_nodes_per_el; ++n) {
                    const stk::mesh::Entity node = elem_nodes[n];
                    const T* x = stk::mesh::field_data(*coord_field, node);
                    // error checking here on x not being nullptr
                    for (int d = 0; d < space_dim; ++d) {
                        h_el_nodes(c, n, d) = x[d];
                    }
                }
            }
            Kokkos::deep_copy(el_coords, h_el_nodes);
            auto ws = ElementBasisWorkset<T, Dim>(basis, geometry_basis, fspace_helper.integration_order, el_coords);
            worksets.push_back(std::move(ws));
        }
    }
}

template class FunctionSpace<double, 1>;
template class FunctionSpace<double, 2>;
template class FunctionSpace<double, 3>;

} // end namespace accopiatti

