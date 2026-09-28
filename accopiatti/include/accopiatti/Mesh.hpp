#pragma once
#include <accopiatti/Types.hpp>
#include <Panzer_STKConnManager.hpp>
#include <Panzer_STK_CubeHexMeshFactory.hpp>
#include <Panzer_STK_CubeTetMeshFactory.hpp>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_SquareQuadMeshFactory.hpp>
#include <Panzer_STK_SquareTriMeshFactory.hpp>
// #include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class Mesh {
public:
    explicit Mesh(stk::ParallelMachine comm, YAML::Node mesh_inputs);

    void add_solution_field(const std::string& field_name, const std::string& block_name) {
        mesh->addSolutionField(field_name, block_name);
    }

    auto create_conn_manager() const {
        return Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    }

    auto get_bulk_data() const {
        return mesh->getBulkData();
    }

    template<Scalar T>
    auto* get_coordinate_field() const {
        return mesh->getMetaData()->get_field<T>(stk::topology::NODE_RANK, "coordinates");
    }

    auto get_dimension() const {
        return mesh->getMetaData()->spatial_dimension();
    }

    std::vector<std::string> get_element_block_names() const {
        std::vector<std::string> block_names = {};
        mesh->getElementBlockNames(block_names);
        return block_names;
    }

    auto get_element_block_topology(const std::string& block_name) const {
        stk::mesh::Part* part = this->get_part(block_name);
        return part->topology();
    }

    std::map<std::string, stk::topology> get_element_block_topologies() const;

    const stk::mesh::BucketVector& get_local_elements(const std::string& block_name) const {
        auto bulk = mesh->getBulkData();
        auto meta = mesh->getMetaData();
        stk::mesh::Part* part = meta->get_part(block_name);
        stk::topology topology = part->topology();
        stk::mesh::Selector selector = 
            stk::mesh::Selector(*part) &
            stk::mesh::Selector(meta->locally_owned_part());
        return bulk->get_buckets(stk::topology::ELEM_RANK, selector);
    }

    auto get_metadata() const {
        return mesh->getMetaData();
    }

    stk::mesh::Part* get_part(const std::string& block_name) const {
        return mesh->getMetaData()->get_part(block_name);
    }

    std::vector<std::string> get_sideset_element_block_names(const std::string& sideset_name) const; 

    stk::mesh::EntityVector get_sideset_node_ids(
        const std::string& sideset_name,
        const std::string& block_name
    ) const {
        stk::mesh::EntityVector nodes;
        mesh->getMyNodes(sideset_name, block_name, nodes);
        return nodes;
    }

    void set_solution_field_data() {
        // mesh->setSolutionFieldData()
    }

    void summarize(std::ostream& stream) const {
        mesh->printMetaData(stream);
    }

    void write_to_exodus(const std::string& file_name) {
        mesh->writeToExodus(file_name);
    }

// private:
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
};

} // end namespace accopiatti
