import accopiatti.cli_parser;
#include <accopiatti/BCs.hpp>
// #include <accopiatti/CLIParser.hpp>
#include <accopiatti/DofManager.hpp>
#include <accopiatti/InputFileParser.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Physics.hpp>
#include <accopiatti/Solver.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        using accopiatti::BCStrategyFactory;
        using accopiatti::CLIParser;
        using accopiatti::DofManager;
        using accopiatti::InputFileParser;
        using accopiatti::Mesh;
        using accopiatti::Physics;
        using accopiatti::LinearSystem;

        stk::ParallelMachine comm = MPI_COMM_WORLD;
        CLIParser clp = CLIParser(comm);
        std::pair<int, int> error_code = clp.parse(argc, argv);
        if (error_code.first != 0) {
            Kokkos::finalize();
            MPI_Finalize();
            return error_code.second;
        }

        std::string input_file = clp.get_option<std::string>("input-file");
        std::string log_file   = clp.get_option<std::string>("log-file");
        Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
        out.setOutputToRootOnly(0);
        out.setShowProcRank(true);

        out << "Input file = " << input_file << std::endl;
        out << "Log file   = " << log_file << std::endl;

        InputFileParser ilp(input_file);
        auto bc_inputs = ilp.get_input_block_raw("boundary conditions");
        auto material_inputs = ilp.get_input_block_raw("materials");
        auto mesh_inputs = ilp.get_input_block_raw("mesh");
        auto physics_inputs = ilp.get_input_block_raw("physics");
        auto solution_inputs = ilp.get_input_block_raw("solution fields");
        out << "Mesh = " << mesh_inputs << std::endl;

        Mesh mesh(comm, mesh_inputs);

        BCStrategyFactory bc_factory;
        std::vector<panzer::BC> bcs = accopiatti::setup_bcs(mesh, bc_inputs);

        DofManager dof_manager(comm, mesh, solution_inputs);
        dof_manager.summarize(out);
        LinearSystem linear_system(comm, dof_manager);

        auto test_workset = accopiatti::build_block_physics_worksets(
            dof_manager, mesh, "block_1", 1024
        );
        std::cout << "Num worksets" << test_workset[0].el_coords(0, 0, 0) << std::endl;

        // auto block_el_ids = mesh.get_block_element_ids("block_1");
        // for (auto local_el : block_el_ids) {
        //     std::vector<accopiatti::GlobalOrdinal> el_gids;
        //     dof_manager.dof_manager->getElementGIDs(local_el, el_gids);
        //     // dof_manager.dof_manager->
        //     std::cout << "Element id = " << local_el << std::endl;
        //     for (auto gid : el_gids) {
        //         std::cout << gid << std::endl;
        //     }
        // }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
