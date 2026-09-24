#pragma once
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include <Intrepid2_HGRAD_HEX_C1_FEM.hpp>
#include <Intrepid2_Orientation.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <stdexcept>

namespace accopiatti {

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

        throw std::runtime_error("Unsupported feature encounted.");
    }

    // should we always be pinging the jacobian
    // off of linear shpae functions?
    // probably a weakness for curved...
    static Teuchos::RCP<Basis<T>> create_geometry(
        stk::topology topology
    ) {
        // return create_h1(topology, 1);
        // if (topology == stk::topology::HEXAHEDRON_8) {
        //     return Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<Device, T, T>());
        // }

        switch (topology) {
        case stk::topology::QUADRILATERAL_4_2D:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<Device, T, T>());
        case stk::topology::HEXAHEDRON_8:
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<Device, T, T>());
        default:
            throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
        }
    }
private:
    static Teuchos::RCP<Basis<T>> create_h1(
        stk::topology topology,
        int order
    ) {
        switch (topology) {
        case stk::topology::QUADRILATERAL_4_2D:
            return Teuchos::rcp(new HgradQuadBasis<T>(order));
        case stk::topology::HEXAHEDRON_8:
            return Teuchos::rcp(new HgradHexBasis<T>(order));
        default:
            throw std::runtime_error("Unsupported topology for Hgrad basis: " + topology.name());
        }

    }

    // TODO implement hcurl, hdiv, L2 wrappers
};

struct FunctionSpaceHelper {
    int basis_order;
    FieldType field_type;
    FunctionSpaceType function_space;
    int integration_order;
};

// TODO only works for total lagrange type implementations currently
template<Scalar T, int Dim>
struct ElementBasisWorkset {
    using CellTools = Intrepid2::CellTools<Device>;
    using FSpaceTools = Intrepid2::FunctionSpaceTools<Device>;

    int num_basis;
    int num_geom_basis;
    int num_elements;
    int num_qp;
    Teuchos::RCP<Basis<T>> basis;
    Teuchos::RCP<Basis<T>> geometry_basis;
    

    CoordinatesView<T, Dim> el_coords;
    Jacobians<T, Dim> jacs;
    JacobianDets<T> jac_dets;
    JacobianInverses<T> jac_invs;
    JxWs<T> jxws;
    PhysicalGradients<T> physical_grads;
    PhysicalPoints<T, Dim> physical_q_pts;
    QuadraturePoints<T> q_pts;
    QuadratureWeights<T> q_wts;
    ReferenceBasisGradients<T> grad_N_xis;
    ReferenceBasisGradients<T> geom_grad_N_xis;
    ReferenceBasisValues<T> Ns;
    ReferenceBasisValues<T> geom_Ns;
    // stk::topology topology;

    ElementBasisWorkset() = default;

    ElementBasisWorkset(
        const Teuchos::RCP<Basis<T>>& basis_,
        const Teuchos::RCP<Basis<T>>& geometry_basis_,
        const int integration_order,
        CoordinatesView<T, Dim>& el_coords_
    );

    void update_jacobians();
    void update_physical_gradients();
    void update_physical_quadrature_points();

    // Add this method to ElementBasisWorkset (or call as a free function passing
    // el_coords / jac_dets) right after update_jacobians() in the constructor.
    // It is deliberately independent of Intrepid2/CellTools: it just reports
    // what raw data went into the Jacobian, so you can tell whether the mesh
    // data itself is bad (duplicate/degenerate nodes) vs. an ordering bug.

    void debug_dump_bad_elements(T tol) const;
};

template<Scalar T, int Dim>
class FunctionSpace {
public:
    explicit FunctionSpace(
        const Mesh& mesh,
        const std::string& block_name,
        const FunctionSpaceHelper& fspace_helper,
        const int workset_size
    );

private:
    std::vector<ElementBasisWorkset<T, Dim>> worksets;
};

} // end namespace accopiatti
