#include <accopiatti/CLIParser.hpp>
#include <accopiatti/Simulation.hpp>

namespace accopiatti {

template<Scalar T>
void run_sim(CLIParser& clp) {

}

template<accopiatti::Scalar T, int Dim>
void run_sim(
    stk::ParallelMachine& comm,
    InputFileParser& ilp,
    Mesh& mesh
) {
    auto bc_inputs = ilp.get_input_block_raw("boundary conditions");
    auto material_inputs = ilp.get_input_block_raw("materials");
    auto physics_inputs = ilp.get_input_block_raw("physics");
    auto solution_inputs = ilp.get_input_block_raw("solution fields");
    DofManager<T, Dim> dof = DofManager<T, Dim>(comm, mesh, solution_inputs);
    // dof.summarize(out);
    // LinearSystem linear_system(comm, dof_manager);
}

template void accopiatti::run_sim<double, 1>(
    stk::ParallelMachine&,
    accopiatti::InputFileParser&,
    accopiatti::Mesh&
);

template void accopiatti::run_sim<double, 2>(
    stk::ParallelMachine&,
    accopiatti::InputFileParser&,
    accopiatti::Mesh&
);

template void accopiatti::run_sim<double, 3>(
    stk::ParallelMachine&,
    accopiatti::InputFileParser&,
    accopiatti::Mesh&
);

} // end namespace accopiatti
