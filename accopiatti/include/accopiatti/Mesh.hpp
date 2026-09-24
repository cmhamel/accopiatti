#pragma once
#include "Panzer_STKConnManager.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
// #include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class Mesh {
public:
    explicit Mesh(stk::ParallelMachine comm, YAML::Node mesh_inputs);

    auto get_conn_manager() {
        return conn_manager;
    }

    auto get_block_element_ids(std::string block_name) {
        return conn_manager->getElementBlock(block_name);
    }

    auto get_dimension() const {
        return mesh->getMetaData()->spatial_dimension();
    }

    std::vector<std::string> get_element_block_names() const {
        std::vector<std::string> block_names = {};
        mesh->getElementBlockNames(block_names);
        return block_names;
    }

    std::map<std::string, stk::topology> get_element_block_topologies() const;

    std::vector<std::string> get_sideset_element_block_names(const std::string& sideset_name) const; 

    stk::mesh::EntityVector get_sideset_node_ids(
        const std::string& sideset_name,
        const std::string& block_name
    ) const {
        stk::mesh::EntityVector nodes;
        mesh->getMyNodes(sideset_name, block_name, nodes);
        return nodes;
    }

// private:
    Teuchos::RCP<panzer_stk::STKConnManager> conn_manager;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
};

} // end namespace accopiatti
