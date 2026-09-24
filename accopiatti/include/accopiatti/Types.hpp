#pragma once
#include <Intrepid2_Basis.hpp>
// #include <Intrepid2_Basis_HGRAD_HEX_Cn_FEM.hpp>
// #include <Intrepid2_HGRAD_HEX_In_FEM.hpp>
#include <Intrepid2_HGRAD_HEX_Cn_FEM.hpp>
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology_decl.hpp>
#include <type_traits>

namespace accopiatti {

// basic types
using Device          = Kokkos::DefaultExecutionSpace;
using EntityId        = stk::mesh::EntityId;
using GlobalOrdinal   = long long;
using HostDevice      = Kokkos::DefaultHostExecutionSpace;
using LocalOrdinal    = int;
// using Scalar          = double;
template<typename T>
concept Scalar        = std::is_arithmetic_v<T>;

// types for bases
template<Scalar T>
using Basis           = Intrepid2::Basis<Device, T, T>;
template<Scalar T>
// using HgradHexBasis   = Intrepid2::Basis
using HgradHexBasis   = Intrepid2::Basis_HGRAD_HEX_Cn_FEM<Device, T, T>;
template<Scalar T>
using HgradQuadBasis  = Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<Device, T, T>;

// types for worksets
template<Scalar T, int Dim>
using CoordinatesView = Kokkos::View<T**[Dim], Device>;

// Quadrature helpers
// all dynamic since these are only run at startup...
// saves some conversions between View and DynRankView
template<Scalar T>
using QuadraturePoints  = Kokkos::DynRankView<T, Device>;
template<Scalar T>
using QuadratureWeights = Kokkos::DynRankView<T, Device>;

// Reference element helpers
// (QP, D)
template<Scalar T>
using ReferenceBasisValues    = Kokkos::DynRankView<T, Device>;
template<Scalar T>
using ReferenceBasisGradients = Kokkos::DynRankView<T, Device>;


// element workset helpers 
template<Scalar T, int Dim>
using Jacobians        = Kokkos::View<T**[Dim][Dim], Device>;
// using Jacobians        = Kokkos::DynRankView<T, Device>;
template<Scalar T>
// using JacobianInverses = Kokkos::View<T**[Dim][Dim], Device>;
using JacobianInverses = Kokkos::DynRankView<T, Device>;
template<Scalar T>
using JacobianDets     = Kokkos::View<T**, Device>;
template<Scalar T>
using JxWs             = Kokkos::View<T**, Device>;

template<Scalar T>
// using PhysicalGradients = Kokkos::View<T***[Dim], Device>;
using PhysicalGradients = Kokkos::DynRankView<T, Device>;
template<Scalar T, int Dim>
using PhysicalPoints = Kokkos::View<T**[Dim], Device>;

// template<int Dim>
// using PointVectorView

// enums
enum class FieldType {
    Scalar,
    SecondOrderTensor,
    SymmetricSecondOrderTensor,
    Vector
};

enum class FunctionSpaceType {
    Hcurl,
    Hdiv,
    Hgrad,
    L2
};

} // end namespace accopiatti
