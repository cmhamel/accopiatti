#include <accopiatti/Parsers.hpp>
#include <accopiatti/Simulation.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <stk_util/parallel/Parallel.hpp>

import accopiatti.physics;
using accopiatti::CLIParser;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        stk::ParallelMachine comm = MPI_COMM_WORLD;
        CLIParser clp = CLIParser(comm);
        std::pair<int, int> error_code = clp.parse(argc, argv);
        if (error_code.first != 0) {
            Kokkos::finalize();
            MPI_Finalize();
            return error_code.second;
        }
        accopiatti::run_sim(comm, clp);
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
