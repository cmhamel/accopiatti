#include <accopiatti/Mesh.hpp>

namespace accopiatti {

Mesh::Mesh(stk::ParallelMachine comm, YAML::Node mesh_inputs) {

    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;

    if (mesh_inputs["type"].as<std::string>() == "structured") {
        Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::rcp(new Teuchos::ParameterList);
        std::string element_type = mesh_inputs["element type"].as<std::string>();

        // setup mesh factory
        if (element_type == "hex") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory);
        } else if (element_type == "quad") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory);
        } else if (element_type == "tet") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory);
        } else if (element_type == "tri") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory);
        } else {
            std::runtime_error("unsupported element type in structured mesh");
        }

        // setup parameters based on dimension
        int x_blocks   = mesh_inputs["x blocks"].as<int>();
        int y_blocks   = mesh_inputs["y blocks"].as<int>();
        int x_elements = mesh_inputs["x elements"].as<int>();
        int y_elements = mesh_inputs["y elements"].as<int>();
        pl->set("X Blocks", x_blocks);
        pl->set("Y Blocks", y_blocks);
        pl->set("X Elements", x_elements);
        pl->set("Y Elements", y_elements);

        if (element_type == "hex" || element_type == "tet") {
            int z_blocks   = mesh_inputs["z blocks"].as<int>();
            int z_elements = mesh_inputs["z elements"].as<int>();
            pl->set("Z Blocks", z_blocks);
            pl->set("Z Elements", z_elements);
        }

        mesh_factory->setParameterList(pl);
        std::runtime_error("support structured mesh type");
    } else if (mesh_inputs["type"].as<std::string>() == "unstructured") {
        std::cout << "Got an unstructured mesh" << std::endl;
        std::string file_name = mesh_inputs["file name"].as<std::string>();
        mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory(file_name));
    } else {
        std::runtime_error("Unsupported mesh type");
    }

    mesh = mesh_factory->buildUncommitedMesh(comm);
    mesh_factory->completeMeshConstruction(*mesh, comm);
    conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
}

std::map<std::string, stk::topology> Mesh::get_element_block_topologies() const {
    std::map<std::string, stk::topology> result;
    auto meta = mesh->getMetaData();

    for (const auto& block_name : get_element_block_names()) {
        stk::mesh::Part* part = meta->get_part(block_name);
        if (part == nullptr) {
            throw std::runtime_error("Could not find element block: " + block_name);
        }
        result.emplace(block_name, part->topology());
    }

    return result;
}

std::vector<std::string> Mesh::get_sideset_element_block_names(const std::string& sideset_name) const {
    std::vector<std::string> result;

    // Make sure the sideset exists.
    if (mesh->getSideset(sideset_name) == nullptr) {
        throw std::runtime_error("Could not find sideset: " + sideset_name);
    }

    // Check every element block.
    for (const auto& block_name : get_element_block_names()) {
        std::vector<stk::mesh::Entity> sides;
        mesh->getAllSides(sideset_name, block_name, sides);
        if (!sides.empty()) {
            result.push_back(block_name);
        }
    }

    return result;
}

} // end namespace accopiatti
