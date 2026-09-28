#pragma once
// Tpetra CSR graph/matrix/vector setup for multi-block FEM problems,
// built on top of accopiatti::DofManager (which itself wraps
// panzer::DOFManager). This intentionally stops at "here is a matrix,
// a rhs, a solution vector, and a place to plug in element kernels" --
// the actual physics (Intrepid2 basis evaluation, residual/jacobian
// kernels) lives in your other code and fills A_overlap_ / b_overlap_
// during the element loop.

#include <Tpetra_Core.hpp>
#include <accopiatti/DofManager.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>

#include <set>
#include <vector>
#include <string>

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Operator.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosLinearProblem.hpp>

#include <Ifpack2_Factory.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayViewDecl.hpp>

namespace accopiatti {

template<Scalar T, int Dim>
class LinearSystem {
public:
    using Node        = typename Tpetra::Map<LocalOrdinal, GlobalOrdinal>::node_type;
    using Map         = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using Graph       = Tpetra::CrsGraph<LocalOrdinal, GlobalOrdinal, Node>;
    using Matrix      = Tpetra::CrsMatrix<T, LocalOrdinal, GlobalOrdinal, Node>;
    using Vector      = Tpetra::Vector<T, LocalOrdinal, GlobalOrdinal, Node>;
    using MultiVector = Tpetra::MultiVector<T, LocalOrdinal, GlobalOrdinal, Node>;
    using Operator    = Tpetra::Operator<T, LocalOrdinal, GlobalOrdinal, Node>;
    using Import      = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
    using Export      = Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node>;

    LinearSystem(
        DofManager<T, Dim>& dof_manager
    ) : comm_(Tpetra::getDefaultComm()), dof_manager_(dof_manager) {
        build_maps();
        build_overlap_graph();
        build_owned_graph();
        build_matrix();
        build_vectors();
    }

    // ---- accessors -------------------------------------------------
    Teuchos::RCP<Matrix>    get_matrix()          { return A_; }
    Teuchos::RCP<Vector>    get_rhs()             { return b_; }
    Teuchos::RCP<Vector>    get_solution()        { return x_; }
    Teuchos::RCP<Matrix>    get_overlap_matrix()  { return A_overlap_; }
    Teuchos::RCP<Vector>    get_overlap_rhs()     { return b_overlap_; }
    Teuchos::RCP<Vector>    get_overlap_solution(){ return x_overlap_; }
    Teuchos::RCP<const Map> get_owned_map() const   { return owned_map_; }
    Teuchos::RCP<const Map> get_overlap_map() const { return overlap_map_; }

    // Pull the current owned solution onto the ghosted layout so that
    // element kernels can see dofs shared with neighboring ranks
    // (needed at the start of every nonlinear iteration / time step).
    void scatter_solution_to_overlap() {
        x_overlap_->doImport(*x_, *importer_, Tpetra::INSERT);
    }

    // Zero the ghosted (element-loop-local) containers before assembly.
    void zero_overlap_containers() {
        A_overlap_->setAllToScalar(Teuchos::ScalarTraits<T>::zero());
        b_overlap_->putScalar(Teuchos::ScalarTraits<T>::zero());
    }

    // Sum the ghosted contributions your kernels wrote into
    // A_overlap_/b_overlap_ back into the uniquely-owned system.
    // Call once per assembly, after the element loop.
    void gather_overlap_to_owned() {
        A_->resumeFill();
        A_->doExport(*A_overlap_, *exporter_, Tpetra::ADD);
        A_->fillComplete();

        b_->putScalar(Teuchos::ScalarTraits<T>::zero());
        b_->doExport(*b_overlap_, *exporter_, Tpetra::ADD);
    }

    // Minimal Belos + Ifpack2 solve. Swap ifpack2_method for "RELAXATION",
    // or replace the preconditioner block with a MueLu hierarchy for
    // elliptic multi-block problems where ILU alone won't scale.
    Belos::ReturnType solve(
        Teuchos::RCP<Teuchos::ParameterList> belos_params,
        const std::string& belos_method = "GMRES",
        const std::string& ifpack2_method = "RILUK"
    ) {
        Ifpack2::Factory factory;
        auto precond = factory.create(ifpack2_method, Teuchos::rcp_dynamic_cast<const Tpetra::RowMatrix<T, LocalOrdinal, GlobalOrdinal, Node>>(A_));
        precond->initialize();
        precond->compute();

        using LinearProblem = Belos::LinearProblem<T, MultiVector, Operator>;
        auto problem = Teuchos::rcp(new LinearProblem(A_, x_, b_));
        problem->setRightPrec(precond);
        problem->setProblem();

        Belos::SolverFactory<T, MultiVector, Operator> solver_factory;
        auto solver = solver_factory.create(belos_method, belos_params);
        solver->setProblem(problem);

        return solver->solve();
    }

private:
    // Owned map: this rank's unique GIDs. Overlap map: owned + ghosted,
    // i.e. every GID this rank touches during an element loop (shared
    // nodes/edges/faces on block or processor boundaries). Both come
    // straight out of the DOFManager, which already merged all fields
    // across every block into one consistent GID numbering.
    void build_maps() {
        std::vector<GlobalOrdinal> owned;
        dof_manager_.get_owned_indices(owned);
        owned_map_ = Teuchos::rcp(new Map(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            Teuchos::arrayViewFromVector(owned),
            0, comm_));

        std::vector<GlobalOrdinal> owned_and_ghosted;
        dof_manager_.get_owned_and_ghosted_indices(owned_and_ghosted);
        overlap_map_ = Teuchos::rcp(new Map(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
            Teuchos::arrayViewFromVector(owned_and_ghosted),
            0, comm_));

        importer_ = Teuchos::rcp(new Import(owned_map_, overlap_map_));
        exporter_ = Teuchos::rcp(new Export(overlap_map_, owned_map_));
    }

    // Sparsity pattern, built on the overlap map so every rank can
    // freely insert its own element-local (row GID x col GID) stencil,
    // including rows it doesn't own. This is the "multi-block" part:
    // we just walk every element block the DOFManager knows about and
    // insert the dense GID x GID stencil per element -- the loop
    // doesn't care that block A might be Hgrad order 2 hexes and block
    // B might be Hgrad order 1 quads, because getElementGIDs already
    // returns the right global numbering for whatever basis/field
    // combination that block was set up with.
    void build_overlap_graph() {
        // Modern Tpetra (post ProfileType/DynamicProfile removal) does
        // NOT grow a graph's allocation on overflow -- the entries-per-
        // row count you give the constructor is a hard preallocation,
        // not a hint. So we do this in two passes: first accumulate the
        // *exact*, deduped column set touched by each overlap-map row
        // using plain STL containers, then construct the graph with
        // the real per-row counts, then insert. (Passing a single
        // scalar like 0, or any one-size-fits-all guess, is exactly
        // what produces "Not enough capacity ... total allocation
        // size 0" the moment a row needs more than that.)
        const LocalOrdinal num_local_rows =
            static_cast<LocalOrdinal>(overlap_map_->getLocalNumElements());
        std::vector<std::set<GlobalOrdinal>> row_columns(num_local_rows);

        std::vector<std::string> block_ids;
        dof_manager_.dof_manager->getElementBlockIds(block_ids);

        for (const auto& block_id : block_ids) {
            const auto& local_elem_ids = dof_manager_.dof_manager->getElementBlock(block_id);
            for (auto local_elem_id : local_elem_ids) {
                std::vector<GlobalOrdinal> gids;
                dof_manager_.dof_manager->getElementGIDs(local_elem_id, gids, block_id);

                for (const auto& row_gid : gids) {
                    const LocalOrdinal row_lid = overlap_map_->getLocalElement(row_gid);
                    row_columns[row_lid].insert(gids.begin(), gids.end());
                }
            }
        }

        Teuchos::ArrayRCP<size_t> num_entries_per_row(num_local_rows);
        for (LocalOrdinal i = 0; i < num_local_rows; ++i) {
            num_entries_per_row[i] = row_columns[i].size();
        }

        // CrsGraph's per-row-count constructor wants a
        // Teuchos::ArrayView<const size_t>, not an ArrayRCP -- the
        // trailing () converts ArrayRCP<const size_t> to that view.
        A_overlap_graph_ = Teuchos::rcp(
            new Graph(overlap_map_, num_entries_per_row.getConst()()));

        for (LocalOrdinal i = 0; i < num_local_rows; ++i) {
            if (row_columns[i].empty()) continue;
            const std::vector<GlobalOrdinal> cols(row_columns[i].begin(), row_columns[i].end());
            const GlobalOrdinal row_gid = overlap_map_->getGlobalElement(i);
            A_overlap_graph_->insertGlobalIndices(
                row_gid, Teuchos::ArrayView<const GlobalOrdinal>(cols));
        }

        // Domain/range = overlap map here; this graph is never solved
        // against directly, it's just scratch space we dedupe below.
        A_overlap_graph_->fillComplete();
    }

    // Combine (dedupe) the overlap sparsity into a graph on the owned
    // map via an ADD-combine export. fillComplete() locks the sparsity
    // pattern so the owned matrix can use a static graph.
    void build_owned_graph() {
        owned_graph_ = Teuchos::rcp(new Graph(owned_map_, 0));
        owned_graph_->doExport(*A_overlap_graph_, *exporter_, Tpetra::ADD);
        owned_graph_->fillComplete();
    }

    void build_matrix() {
        A_         = Teuchos::rcp(new Matrix(owned_graph_));
        A_overlap_ = Teuchos::rcp(new Matrix(A_overlap_graph_));
    }

    void build_vectors() {
        b_          = Teuchos::rcp(new Vector(owned_map_));
        x_          = Teuchos::rcp(new Vector(owned_map_));
        b_overlap_  = Teuchos::rcp(new Vector(overlap_map_));
        x_overlap_  = Teuchos::rcp(new Vector(overlap_map_));
    }

    Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    DofManager<T, Dim>& dof_manager_;

    Teuchos::RCP<const Map> owned_map_;
    Teuchos::RCP<const Map> overlap_map_;
    Teuchos::RCP<Import>    importer_;
    Teuchos::RCP<Export>    exporter_;

    Teuchos::RCP<Graph> A_overlap_graph_;
    Teuchos::RCP<Graph> owned_graph_;

    Teuchos::RCP<Matrix> A_;
    Teuchos::RCP<Matrix> A_overlap_;
    Teuchos::RCP<Vector> b_;
    Teuchos::RCP<Vector> x_;
    Teuchos::RCP<Vector> b_overlap_;
    Teuchos::RCP<Vector> x_overlap_;
};

} // namespace accopiatti
