#pragma once
#include <accopiatti/BCs.hpp>
#include <accopiatti/DofManager.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Parsers.hpp>
#include <accopiatti/PoissonAssembler.hpp>
#include <accopiatti/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

namespace accopiatti {

struct Simulation {
    InputFileParser ifp;
    Mesh mesh;
    std::string input_file;
    std::string log_file;
    Teuchos::FancyOStream out;

    // // Simulation() = default;
    // explicit Simulation(stk::ParallelMachine comm, CLIParser& clp) 
    //     :
    //     input_file(clp.get_option<std::string>("input-file")),
    //     log_file(clp.get_option<std::string>("log-file")),
    //     ifp(InputFileParser(input_file)),
    //     out(Teuchos::rcpFromRef(std::cout)) {
    //     out.setOutputToRootOnly(0);
    //     out.setShowProcRank(true);

    //     out << "Input file = " << input_file << std::endl;
    //     out << "Log file   = " << log_file << std::endl;

    //     // ilp = InputFileParser(input_file);
    //     mesh = Mesh(comm, ifp.get_input_block_raw("mesh"));
    //     auto block_names = mesh.get_element_block_names();
    //     for (auto& block_name : block_names) {
    //         std::cout << block_name << std::endl;
    //     }
    // }
};

static FieldType parse_field_type(const std::string& name) {
    if (name == "scalar") {
        return FieldType::Scalar;
    } else if (name == "second order tensor") {
        return FieldType::SecondOrderTensor;
    } else if (name == "symmetric second order tensor") {
        return FieldType::SymmetricSecondOrderTensor;
    } else if (name == "vector") {
        return FieldType::Vector;
    } else {
        throw std::runtime_error("Unsupported field type: " + name);
    }
}

static FunctionSpaceType parse_function_space(const std::string& name) {
    if (name == "Hcurl") {
        return FunctionSpaceType::Hcurl;
    } else if (name == "Hdiv") {
        return FunctionSpaceType::Hdiv;
    } else if (name == "Hgrad") {
        return FunctionSpaceType::Hgrad;
    } else if (name == "L2") {
        return FunctionSpaceType::L2;
    } else {
        throw std::runtime_error("Unsupported function space: " + name);
    }
}

static std::map<std::string, FieldSetupHelper> parse_fields(const InputFileParser& ifp) {
    std::map<std::string, FieldSetupHelper> field_helpers;
    auto solution_inputs = ifp.get_input_block_raw("solution fields");
    for (const auto& field : solution_inputs) {
            const auto name = field.first.as<std::string>();
            const auto config = field.second;

            field_helpers.emplace(
                name,
                FieldSetupHelper{
                    config["basis order"].as<int>(),
                    parse_field_type(config["field type"].as<std::string>()),
                    parse_function_space(config["function space"].as<std::string>()),
                    config["integration order"].as<int>()
                }
            );
        }
    return field_helpers;
}

void run_sim(stk::ParallelMachine& comm, CLIParser& clp);

template<Scalar T, int Dim>
void run_sim_inner(
    stk::ParallelMachine& comm,
    InputFileParser& ifp,
    Mesh& mesh
) {
    auto solution_inputs = parse_fields(ifp);
    DofManager<T, Dim> dof_manager = DofManager<T, Dim>(comm, mesh, solution_inputs);

    switch(dof_manager.get_num_solution_fields()) {
        case 1:
            run_sim_inner_inner<T, Dim, 1>(comm, ifp, mesh, dof_manager, solution_inputs);
            return;
        case 2:
            run_sim_inner_inner<T, Dim, 2>(comm, ifp, mesh, dof_manager, solution_inputs);
            return;
        case 3:
            run_sim_inner_inner<T, Dim, 3>(comm, ifp, mesh, dof_manager, solution_inputs);
            return;
        default:
            std::cout << "Solution field number = " << dof_manager.get_num_solution_fields() << std::endl;
            throw std::runtime_error("Got too large a number of solution fields. File an issue.");
    }
}

template<Scalar T, int Dim, int NumSol>
void run_sim_inner_inner(
    stk::ParallelMachine& comm,
    InputFileParser& ifp,
    Mesh& mesh,
    DofManager<T, Dim>& dof_manager,
    std::map<std::string, FieldSetupHelper>& field_helpers
) {
    const int workset_size = 1024; // change this based on backend
    auto bc_inputs = ifp.get_input_block_raw("boundary conditions");
    auto dirichlet_bcs = setup_dirichlet_bcs(mesh, dof_manager, field_helpers, bc_inputs);
    auto fspace = FunctionSpace<T, Dim, NumSol>(mesh, dof_manager, workset_size);
    // fspace.block_worksets[1].geom_workset.debug_dump_bad_elements(1e-6);
    auto linear_system = LinearSystem<T, Dim>(dof_manager);

    // assembler
    auto assembler = PoissonAssembler<T, Dim, NumSol>(fspace, linear_system);
    assembler.assemble();
    apply_homogeneous_dirichlet(linear_system, dirichlet_bcs.global_ids);

    // assembler.
    auto belos_params = Teuchos::rcp(new Teuchos::ParameterList("Belos"));
    belos_params->set("Convergence Tolerance", 1e-12);
    belos_params->set("Maximum Iterations", 500);
    belos_params->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::FinalSummary);

    // Poisson's matrix is SPD -- CG converges faster and cheaper per
    // iteration than GMRES here. RILUK (ILU) is a fine general-purpose
    // preconditioner even though it's not strictly symmetric; if you want
    // a symmetric one to pair more naturally with CG, "RELAXATION" with
    // symmetric Gauss-Seidel is a common choice too.
    auto result = linear_system.solve(belos_params, "CG", "RELAXATION");

    auto x = linear_system.get_solution();
    T local_max = x->getVector(0)->normInf();  // or a manual Kokkos reduce_max on the host mirror
    std::cout << "max |u| on owned vector (pre-Exodus) = " << local_max << std::endl;

    if (result != Belos::Converged) {
        std::cerr << "Warning: solver did not converge\n";
    }

    write_solution_field<T, Dim, NumSol>(mesh, fspace, linear_system, "u");
    mesh.write_to_exodus("test.exo");
}

template<Scalar T, int Dim, int NumSol>
void write_solution_field(
    Mesh& mesh,
    FunctionSpace<T, Dim, NumSol>& function_space,
    LinearSystem<T, Dim>& linear_system,
    const std::string& field_name
) {
    linear_system.scatter_solution_to_overlap();
    // auto h_x_overlap = Kokkos::create_mirror_view_and_copy(
    //     Kokkos::HostSpace(), *linear_system.get_overlap_solution()
    // );
    auto x_overlap = linear_system.get_overlap_solution();
    auto h_x_overlap = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        x_overlap->getLocalViewHost(Tpetra::Access::ReadOnly)
    );
    auto overlap_map = linear_system.get_overlap_map();
 
    for (const auto& block_ws : function_space.get_block_worksets()) {
        const auto& sol_ws = block_ws.sol_worksets[0];
        const int num_elements = sol_ws.num_elements;
        const int num_basis    = sol_ws.num_basis;
 
        auto h_gids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), block_ws.global_dof_ids);
        auto h_local_elem_ids = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), block_ws.local_el_ids
        );
 
        // setSolutionFieldData loops i = 0..(mesh topology's node count)-1
        // and reads values(cell, i) -- if the solution basis has FEWER
        // basis functions than the mesh element has nodes, that reads past
        // the end of this view. Geometry in this codebase is always linear
        // (4/8 nodes), so this should never trip in practice, but it's a
        // one-line guard against a genuinely dangerous silent OOB read.
        // (num_basis > num_mesh_nodes is fine -- see the accompanying note
        // on why higher-order fields degrade gracefully instead.)
        const size_t num_mesh_nodes = mesh.get_element_block_topology(block_ws.block_name).num_nodes();
        TEUCHOS_TEST_FOR_EXCEPTION(
            static_cast<size_t>(num_basis) < num_mesh_nodes, std::runtime_error,
            "write_solution_field: solution basis (" << num_basis << " dofs) has fewer "
            "entries than block \"" << block_ws.block_name << "\"'s mesh topology has nodes ("
            << num_mesh_nodes << "). setSolutionFieldData would read out of bounds.");
 
        // setSolutionFieldData does its own create_mirror_view/deep_copy
        // internally and indexes as (cell, localNodeIndex) -- build that
        // shape directly so no extra copy is needed on its end.
        Kokkos::View<T**, Kokkos::HostSpace> values("solution_values", num_elements, num_basis);
        std::vector<std::size_t> local_elem_ids(num_elements);
 
        for (int c = 0; c < num_elements; ++c) {
            local_elem_ids[c] = h_local_elem_ids(c);
            for (int i = 0; i < num_basis; ++i) {
                const GlobalOrdinal gid = h_gids(c, i);
                const LocalOrdinal lid = overlap_map->getLocalElement(gid);
                values(c, i) = h_x_overlap(lid, 0);
            }
        }
 
        mesh.mesh->setSolutionFieldData(field_name, block_ws.block_name, local_elem_ids, values);
    }
}


} // end namespace accopiatti
