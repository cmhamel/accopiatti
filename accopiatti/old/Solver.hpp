#pragma once
#include <accopiatti/DofManager.hpp>
#include <Teuchos_RCP.hpp>
// #include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_FECrsMatrix.hpp>
#include <Tpetra_FEMultiVector.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

namespace accopiatti {

class LinearSystem {
public:
    using Scalar = double;
    using Node = Tpetra::Map<LocalOrdinal, GlobalOrdinal>::node_type;
    using Import = Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node>;
    using Map = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using Vector = Tpetra::FEMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    // using Vector = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using Matrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    explicit LinearSystem(
        stk::ParallelMachine comm_,
        const DofManager& dof_manager
    ) {
        auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(comm_));
        std::vector<GlobalOrdinal> owned_gids = {};
        std::vector<GlobalOrdinal> owned_and_ghosted_gids = {};
        dof_manager.get_owned_indices(owned_gids);
        dof_manager.get_owned_and_ghosted_indices(owned_and_ghosted_gids);
        Teuchos::ArrayView<const GlobalOrdinal> owned_view(
            owned_gids.data(), owned_gids.size()
        );
        Teuchos::ArrayView<const GlobalOrdinal> owned_and_ghosted_view(
            owned_and_ghosted_gids.data(), owned_and_ghosted_gids.size()
        );
        // auto overlap_gids = dof_manager.get_owned_and_ghosted_gids();
        Teuchos::RCP<Map> owned_map = Teuchos::rcp(
            new Map(
                Teuchos::OrdinalTraits<GlobalOrdinal>::invalid(),
                owned_view,
                0,
                comm
            )
        );
        Teuchos::RCP<Map> owned_and_ghosted_map = Teuchos::rcp(
            new Map(
                Teuchos::OrdinalTraits<GlobalOrdinal>::invalid(),
                owned_and_ghosted_view,
                0,
                comm
            )
        );
        auto importer = Teuchos::rcp(new Import(owned_map, owned_and_ghosted_map));

        const size_t estimated_entries_per_row = 81;
        // auto A = Teuchos::rcp(new Tpetra::FECrsMatrix(owned_map));
        // rhs = Teuchos::rcp(new Vector(map));
        // auto b = Teuchos::rcp(new Tpetra::FEMultiVector(owned_map, 1));
        // solution = Teuchos::rcp(new Vector(map));
        // auto A = Teuchos::rcp(new Matrix(owned_map));
        auto b = Teuchos::rcp(new Vector(owned_map, importer, 1));
    }

    // void assemble_element_residual(
    //     Vector& R,
    //     const std::vector<GlobalOrdinal> 
    // );

private:
    // Teuchos::RCP<const Map> owned_map;
    // Teuchos::RCP<const Map> owned_and_ghosted_map;
    // Teuchos::RCP<Vector> rhs;
    // Teuchos::RCP<Vector> solution;
};

} // end namespace accopiatti
