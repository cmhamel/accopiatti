#pragma once
#include <accopiatti/CLIParser.hpp>
#include <accopiatti/DofManager.hpp>
#include <accopiatti/InputFileParser.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

namespace accopiatti {

struct Simulation {
    InputFileParser ifp;
    Mesh mesh;
    std::string input_file;
    std::string log_file;
    Teuchos::FancyOStream out;

    // Simulation() = default;
    explicit Simulation(stk::ParallelMachine comm, CLIParser& clp) 
        :
        input_file(clp.get_option<std::string>("input-file")),
        log_file(clp.get_option<std::string>("log-file")),
        ifp(InputFileParser(input_file)),
        out(Teuchos::rcpFromRef(std::cout)) {
        out.setOutputToRootOnly(0);
        out.setShowProcRank(true);

        out << "Input file = " << input_file << std::endl;
        out << "Log file   = " << log_file << std::endl;

        // ilp = InputFileParser(input_file);
        mesh = Mesh(comm, ifp.get_input_block_raw("mesh"));
        auto block_names = mesh.get_element_block_names();
        for (auto& block_name : block_names) {
            std::cout << block_name << std::endl;
        }
    }
};

template<Scalar T, int Dim>
void run_sim(
    stk::ParallelMachine& comm,
    InputFileParser& ilp,
    Mesh& mesh
);

} // end namespace accopiatti
