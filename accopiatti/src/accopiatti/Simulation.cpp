#include <accopiatti/FunctionSpace.hpp>
#include <accopiatti/LinearSystem.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Parsers.hpp>
#include <accopiatti/Simulation.hpp>
#include <stk_util/parallel/Parallel.hpp>

namespace accopiatti {

void run_sim(stk::ParallelMachine& comm, CLIParser& clp) {
    std::string input_file = clp.get_option<std::string>("input-file");
    std::string log_file   = clp.get_option<std::string>("log-file");
    Teuchos::FancyOStream out(Teuchos::rcpFromRef(std::cout));
    out.setOutputToRootOnly(0);
    out.setShowProcRank(true);

    InputFileParser ilp(input_file);
    auto mesh_inputs = ilp.get_input_block_raw("mesh");

    Mesh mesh(comm, mesh_inputs);
    const int dim = mesh.get_dimension();

    switch (dim) {
        case 1:
            run_sim_inner<double, 1>(comm, ilp, mesh);
            return;
        case 2:
            run_sim_inner<double, 2>(comm, ilp, mesh);
            return;
        case 3:
            run_sim_inner<double, 3>(comm, ilp, mesh);
            return;
        default:
            throw std::runtime_error("Got dimension other than 1, 2, or 3.");
    }
}

} // end namespace accopiatti
