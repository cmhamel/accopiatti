#pragma once
// -Δu = f on the unit square, u = 0 on the boundary.
// f(x,y) = 2*pi^2*sin(pi x)*sin(pi y) gives the exact solution
// u(x,y) = sin(pi x)*sin(pi y) -- the standard Poisson sanity check.
// This only does the FILL (stiffness + load into the ghosted/overlap
// containers, then gathered to owned); Dirichlet BCs and the actual
// solve are separate steps, noted at the bottom.

#include <accopiatti/DofManager.hpp>
#include <accopiatti/FunctionSpace.hpp>
#include <accopiatti/LinearSystem.hpp>
#include <accopiatti/Types.hpp>

#include <Kokkos_Core.hpp>
#include <Teuchos_ArrayView.hpp>

#include <cmath>
#include <vector>

namespace accopiatti {

template<Scalar T>
inline T poisson_forcing(T x, T y) {
    constexpr T pi = static_cast<T>(M_PI);
    return static_cast<T>(2.0) * pi * pi * std::sin(1.0 * pi * x) * std::sin(1.0 * pi * y);
    // return static_cast<T>(1.0);
}

// Single scalar field (NumSol = 1) Poisson assembler. BlockWorkset's
// global_ids is [num_elements, num_block_dofs] with every field's
// dofs concatenated -- for NumSol > 1 you'd slice out this field's
// offset/width within that second dimension. Poisson only registers
// one field per block, so offset 0 / width num_basis lines up exactly.
template<Scalar T, int Dim, int NumSol>
class PoissonAssembler {
public:
    static constexpr int SolId = 0;

    PoissonAssembler(
        FunctionSpace<T, Dim, NumSol>& function_space,
        LinearSystem<T, Dim>& linear_system
    ) : function_space_(function_space), linear_system_(linear_system) {}

    void assemble() {
        linear_system_.zero_overlap_containers();

        for (const auto& block_ws : function_space_.get_block_worksets()) {
            assemble_block_workset(block_ws);
        }

        linear_system_.gather_overlap_to_owned();
    }

private:
    // Host loops on purpose: worksets are small chunks (workset_size
    // elements), so this is a demo of *what* to compute, not the fast
    // path. Once this is verified, swap the inner (i, j, p) loops for
    // a Kokkos::parallel_for over sol_ws.physical_grads directly and
    // do the global scatter afterward -- element-local math doesn't
    // change, only where it runs.
    void assemble_block_workset(const BlockWorkset<T, Dim, NumSol>& block_ws) {
        const auto& geom_ws = block_ws.geom_workset;
        const auto& sol_ws  = block_ws.sol_worksets[SolId];

        auto h_jxw   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), geom_ws.jxws);
        auto h_grads = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol_ws.physical_grads);
        auto h_Ns    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol_ws.Ns);
        auto h_qpts  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol_ws.physical_q_pts);
        auto h_gids  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), block_ws.global_dof_ids);

        const int num_basis = sol_ws.num_basis;
        const int num_qp    = sol_ws.num_qp;

        std::vector<T> Ke(static_cast<size_t>(num_basis) * num_basis);
        std::vector<T> Fe(num_basis);
        std::vector<GlobalOrdinal> gids(num_basis);

        auto A_overlap = linear_system_.get_overlap_matrix();
        auto b_overlap = linear_system_.get_overlap_rhs();

        for (int c = 0; c < sol_ws.num_elements; ++c) {
            std::fill(Ke.begin(), Ke.end(), T(0));
            std::fill(Fe.begin(), Fe.end(), T(0));
            for (int i = 0; i < num_basis; ++i) {
                // NumSol == 1, so this field's dofs are the whole row.
                gids[i] = h_gids(c, i);
            }

            for (int p = 0; p < num_qp; ++p) {
                const T jxw = h_jxw(c, p);
                const T x   = h_qpts(c, p, 0);
                const T y   = h_qpts(c, p, 1);
                const T f   = poisson_forcing<T>(x, y);

                for (int i = 0; i < num_basis; ++i) {
                    Fe[i] += h_Ns(i, p) * f * jxw;
                    for (int j = 0; j < num_basis; ++j) {
                        T dot = T(0);
                        for (int d = 0; d < Dim; ++d) {
                            dot += h_grads(c, i, p, d) * h_grads(c, j, p, d);
                        }
                        Ke[static_cast<size_t>(i) * num_basis + j] += dot * jxw;
                    }
                }
            }

            for (int i = 0; i < num_basis; ++i) {
                Teuchos::ArrayView<const GlobalOrdinal> col_view(gids);
                Teuchos::ArrayView<const T> row_view(&Ke[static_cast<size_t>(i) * num_basis], num_basis);
                A_overlap->sumIntoGlobalValues(gids[i], col_view, row_view);
                b_overlap->sumIntoGlobalValue(gids[i], Fe[i]);
            }
        }
    }

    FunctionSpace<T, Dim, NumSol>& function_space_;
    LinearSystem<T, Dim>& linear_system_;
};

// -----------------------------------------------------------------
// Homogeneous Dirichlet BC on a given set of owned boundary GIDs:
// zero the row in the owned matrix, put 1 on the diagonal, zero the
// rhs entry. Call AFTER gather_overlap_to_owned(), before solve().
// Getting the boundary GID list is mesh-specific (side set, or a
// coordinate check on owned dofs); left out here since it depends on
// what Mesh exposes, but once you have the GIDs this part is generic:
// -----------------------------------------------------------------

template<Scalar T, int Dim>
void apply_homogeneous_dirichlet(
    LinearSystem<T, Dim>& linear_system,
    const std::vector<GlobalOrdinal>& boundary_gids
) {
    auto A = linear_system.get_matrix();
    auto b = linear_system.get_rhs();
    auto owned_map = linear_system.get_owned_map();
 
    using Matrix = typename LinearSystem<T, Dim>::Matrix;
    using GidView = typename Matrix::nonconst_global_inds_host_view_type;
    using ValView = typename Matrix::nonconst_values_host_view_type;
 
    A->resumeFill();
    for (const auto& gid : boundary_gids) {
        if (!owned_map->isNodeGlobalElement(gid)) continue;
 
        const size_t num_entries = A->getNumEntriesInGlobalRow(gid);
        GidView cols("dirichlet_row_cols", num_entries);
        ValView vals("dirichlet_row_vals", num_entries);
        size_t num_returned = 0;
        A->getGlobalRowCopy(gid, cols, vals, num_returned);
 
        std::vector<T> zero_vals(num_returned, T(0));
        std::vector<GlobalOrdinal> host_cols(num_returned);
        for (size_t k = 0; k < num_returned; ++k) {
            host_cols[k] = cols(k);
        }
        A->replaceGlobalValues(gid, Teuchos::ArrayView<const GlobalOrdinal>(host_cols),
                                Teuchos::ArrayView<const T>(zero_vals));
        A->replaceGlobalValues(gid, Teuchos::ArrayView<const GlobalOrdinal>(&gid, 1),
                                Teuchos::ArrayView<const T>(std::vector<T>{T(1)}));
 
        b->replaceGlobalValue(gid, T(0));
    }
    A->fillComplete();
}


} // namespace accopiatti
