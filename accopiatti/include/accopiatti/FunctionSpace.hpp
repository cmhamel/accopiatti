#pragma once
#include <accopiatti/Types.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>

namespace accopiatti {

// Intrepid2's Basis::getValues, Cubature::getCubature, CellTools and
// FunctionSpaceTools are all written against Kokkos::DynRankView. Our storage is
// static-rank (fast in kernels), so wrap it in a non-owning-semantics alias
// (shares the same allocation, no copy) at the call boundary.
template<Scalar T, class ViewType>
inline Kokkos::DynRankView<T, Device> as_dyn(const ViewType& v) {
    return Kokkos::DynRankView<T, Device>(v);
}

template<Scalar T>
class BasisFactory {
public:
    static Teuchos::RCP<Basis<T>> create(
        stk::topology topology,
        FunctionSpaceType function_space,
        int order
    ) {
        switch (function_space) {
        case FunctionSpaceType::Hcurl:
            throw std::runtime_error("Finish Hcurl wrapper methods");
        case FunctionSpaceType::Hdiv:
            throw std::runtime_error("Finish Hdiv wrapper methods");
        case FunctionSpaceType::Hgrad:
            return create_h1(topology, order);
        case FunctionSpaceType::L2:
            throw std::runtime_error("Finish L2 wrapper methods");
        }
    }

    // should we always be pinging the jacobian
    // off of linear shpae functions?
    // probably a weakness for curved...
    static Teuchos::RCP<Basis<T>> create_geometry(
        stk::topology topology
    ) {
        return create_h1(topology, 1);
    }
private:
    static Teuchos::RCP<Basis<T>> create_h1(
        stk::topology topology,
        int order
    ) {
        if (topology == stk::topology::QUADRILATERAL_4_2D) {
            return Teuchos::rcp(new HgradQuadBasis<T>(order));
        }

        if (topology == stk::topology::HEXAHEDRON_8) {
            return Teuchos::rcp(new HgradHexBasis<T>(order));
        }

        throw std::runtime_error("Unsupported topology for H1 basis: " + topology.name());
    }

    // TODO implement hcurl, hdiv, L2 wrappers
};

struct FunctionSpaceHelper {
    int basis_order;
    FieldType field_type;
    FunctionSpaceType function_space;
    int integration_order;
};

template<Scalar T, int Dim>           // D
struct ReferenceElement {
    int num_basis = 0;      // F
    int num_geo_nodes = 0;  // N
    int num_qp = 0;         // Q
    QuadraturePoints<T> q_pts; // (Q, D)
    QuadratureWeights<T> q_wts;     // (Q,)
    ReferenceValues<T> Ns;          // (F, Q)
    ReferenceGradients<T> grad_N_xis; // (F, Q, D)
};

template<Scalar T, int Dim>
struct ElementBasisWorkset {
    Jacobians<T, Dim> jacs;
    JacobianDets<T> jac_dets;
    JacobianInverses<T, Dim> jac_invs;
    JxWs<T> JxWs;
    PhysicalGradients<T, Dim> grad_Ns;
};


template<Scalar T, int Dim>
struct FunctionSpaceWorkset {
    std::string block_name;
    stk::topology topology;
    int num_els = 0;
    int num_nodes_per_el = 0;
    int space_dim = 0;
    CoordinatesView<T> el_coords;
    ElementBasisWorkset<T, Dim> workset;
};

// TODO need to add a lot of error checking
template<Scalar T, int Dim>
ReferenceElement<T, Dim> build_reference_data(
    const Teuchos::RCP<Basis<T>>& basis,
    const Teuchos::RCP<Basis<T>>& geometry_basis,
    const int integration_order
) {
    ReferenceElement<T, Dim> ref;
    const shards::CellTopology cell_topo = basis->getBaseCellTopology();
    auto quadrature = Intrepid2::DefaultCubatureFactory::create<Device, T, T>(cell_topo, integration_order);

    ref.num_basis = static_cast<int>(basis->getCardinality());
    ref.num_geo_nodes = static_cast<int>(basis->getCardinality());
    ref.num_qp = static_cast<int>(quadrature->getNumPoints());
    const int F = ref.num_basis;
    const int N = ref.num_geo_nodes;
    const int Q = ref.num_qp;

    // setup quadrature
    ref.q_pts = QuadraturePoints<T>("reference_quadrature_points", Q, Dim);
    ref.q_wts = QuadratureWeights<T>("reference_quadrature_weights", Q);
    quadrature->getCubature(ref.q_pts, ref.q_wts);

    // // setup basis values
    ref.Ns = ReferenceValues<T>("reference_basis_values", F, Q);
    ref.grad_N_xis = ReferenceGradients<T>("reference_basis_gradients", F, Q, Dim);
    basis->getValues(ref.Ns, ref.q_pts, Intrepid2::OPERATOR_VALUE);
    basis->getValues(ref.grad_N_xis, ref.q_pts, Intrepid2::OPERATOR_GRAD);
    return ref;
}

template<Scalar T, int Dim>
ElementBasisWorkset<T, Dim> build_element_workset(
    FunctionSpaceWorkset<T, Dim>& workset,
    const ReferenceElement<T, Dim>& ref,
    const Teuchos::RCP<Basis<T>>& geometry_basis
) {
    using CellTools = Intrepid2::CellTools<Device>;
    using FSpaceTools = Intrepid2::FunctionSpaceTools<Device>;
    const int C = workset.num_els;
    const int F = ref.num_basis;
    const int Q = ref.num_qp;
    ElementBasisWorkset<T, Dim> el_workset;

    Jacobians<T, Dim> jacs("jacobians", C, Q);
    el_workset.jacs = Jacobians<T, Dim>("jacobians", C, Q);
    el_workset.jac_invs = JacobianInverses<T, Dim>("jacobian_inverse", C, Q);
    el_workset.jac_dets = JacobianInverses<T, Dim>("jacobian_dets", C, Q);
    el_workset.JxWs = JxWs<T>("JxWs", C, Q);

    CellTools::setJacobian(el_workset.jacs, ref.q_pts, el_workset.el_coords, geometry_basis);
    CellTools::setJacobianInv(el_workset.jac_invs, el_workset.jacs);
    CellTools::setJacobianDet(el_workset.jac_dets, el_workset.jacs);
}

template<Scalar T, int Dim>
class FunctionSpace {
public:
    explicit FunctionSpace(
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
        basis = BasisFactory<T>::create(topology, fspace_helper.function_space, fspace_helper.basis_order);
        geometry_basis = BasisFactory<T>::create_geometry(topology);

        const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
        if (num_nodes_per_el != static_cast<int>(geometry_basis->getCardinality())) {
            throw std::runtime_error("FunctionSpace: STK node count != geometry basis cardinality");
        }

        ref_fe = build_reference_data<T, Dim>(basis, geometry_basis, fspace_helper.integration_order);
        // auto* coord_field = meta->get_field<T>(stk::topology::NODE_RANK, "coordinates");
        // stk::mesh::Selector selector = 
        //     stk::mesh::Selector(*block_part) &
        //     stk::mesh::Selector(meta->locally_owned_part());
        // const stk::mesh::BucketVector& buckets = bulk->get_buckets(stk::topology::ELEM_RANK, selector);
        // std::vector<FunctionSpaceWorkset<T, Dim>> worksets;
        
        // for (const stk::mesh::Bucket* bucket : buckets) {
        //     const stk::topology topology = bucket->topology();
        //     const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
        //     const int bucket_size = static_cast<int>(bucket->size());
        //     for (int bucket_offset = 0; bucket_offset < bucket_size; bucket_offset += workset_size) {
        //         const int num_els_in_workset = std::min(workset_size, bucket_size - bucket_offset);
        //         FunctionSpaceWorkset<T, Dim> ws = FunctionSpaceWorkset<T, Dim> {
        //             block_name, topology, num_els_in_workset, num_nodes_per_el, space_dim,
        //             CoordinatesView<T>("el_nodes_" + block_name, num_els_in_workset, num_nodes_per_el, space_dim)
        //         };
        //         auto h_el_nodes = Kokkos::create_mirror_view(ws.el_coords);
        //         for (int c = 0; c < num_els_in_workset; ++c) {
        //             const stk::mesh::Entity elem = (*bucket)[bucket_offset + c];
        //             const unsigned elem_num_nodes = bulk->num_nodes(elem);
        //             // error check this
        //             const stk::mesh::Entity* elem_nodes = bulk->begin_nodes(elem);

        //             for (int n = 0; n < num_nodes_per_el; ++n) {
        //                 const stk::mesh::Entity node = elem_nodes[n];
        //                 const T* x = stk::mesh::field_data(*coord_field, node);
        //                 // error checking here on x not being nullptr
        //                 for (int d = 0; d < space_dim; ++d) {
        //                     h_el_nodes(c, n, d) = x[d];
        //                 }
        //             }
        //         }
        //         Kokkos::deep_copy(ws.el_coords, h_el_nodes);
        //         build_element_workset<T, Dim>(ws, ref_fe, geometry_basis);
        //         worksets.push_back(std::move(ws));
        //     }
        // }
    }
private:
    Teuchos::RCP<Basis<T>> basis;
    Teuchos::RCP<Basis<T>> geometry_basis;
    ReferenceElement<T, Dim> ref_fe;
    // std::vector<FunctionSpaceWorkSet> worksets;
};


} // end namespace accopiatti
