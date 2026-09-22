#pragma once
// #include "Panzer_EquationSet_DefaultImpl.hpp"
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class Physics {
public:
    explicit Physics(
        YAML::Node material_inputs,
        YAML::Node solution_inputs
    ) {
        auto block_assignments = material_inputs["block assignment"]
            .as<std::map<std::string, std::string>>();
        auto models = material_inputs["models"];

        // if (block_assignments.
        // if (material_inputs.
        // // auto fields = solution_inputs
        // std::cout << "Block assignments = " << models << std::endl;
    }
private:
};

template<class BasisType, int Dim>
class BlockPhysics {
public:
    explicit BlockPhysics(BasisType basis) {
        Kokkos::View<double***, Device> cell_nodes("cell_nodes", 1, basis->getCardinality(), Dim);
    }

};

struct BlockPhysicsWorkset {
    std::string block_name;
    stk::topology topology;
    int num_els = 0;
    int num_nodes_per_el = 0;
    int space_dim = 0;
    CoordinatesView<double> el_coords;
};

std::vector<BlockPhysicsWorkset> build_block_physics_worksets(
    DofManager& dof_manager,
    Mesh& mesh,
    std::string block_name,
    const int workset_size
) {
    // error check on workset_size
    // auto coords = mesh.mesh->
    // auto interps = dof_manager.get_basis()
    auto bulk = mesh.mesh->getBulkData();
    auto meta = mesh.mesh->getMetaData();
    const int space_dim = static_cast<int>(meta->spatial_dimension());
    stk::mesh::Part* block_part = meta->get_part(block_name);
    auto* coord_field = meta->get_field<double>(stk::topology::NODE_RANK, "coordinates");
    stk::mesh::Selector selector = 
        stk::mesh::Selector(*block_part) &
        stk::mesh::Selector(meta->locally_owned_part());
    const stk::mesh::BucketVector& buckets = bulk->get_buckets(stk::topology::ELEM_RANK, selector);
    std::vector<BlockPhysicsWorkset> worksets;

    for (const stk::mesh::Bucket* bucket : buckets) {
        const stk::topology topology = bucket->topology();
        const int num_nodes_per_el = static_cast<int>(topology.num_nodes());
        const int bucket_size = static_cast<int>(bucket->size());
        for (int bucket_offset = 0; bucket_offset < bucket_size; bucket_offset += workset_size) {
            const int num_els_in_workset = std::min(workset_size, bucket_size - bucket_offset);
            BlockPhysicsWorkset ws = BlockPhysicsWorkset {
                block_name, topology, num_els_in_workset, num_nodes_per_el, space_dim,
                CoordinatesView<double>("el_nodes_" + block_name, num_els_in_workset, num_nodes_per_el, space_dim)
            };
            auto h_el_nodes = Kokkos::create_mirror_view(ws.el_coords);
            for (int c = 0; c < num_els_in_workset; ++c) {
                const stk::mesh::Entity elem = (*bucket)[bucket_offset + c];
                const unsigned elem_num_nodes = bulk->num_nodes(elem);
                // error check this
                const stk::mesh::Entity* elem_nodes = bulk->begin_nodes(elem);

                for (int n = 0; n < num_nodes_per_el; ++n) {
                    const stk::mesh::Entity node = elem_nodes[n];
                    const double* x = stk::mesh::field_data(*coord_field, node);
                    // error checking here on x not being nullptr
                    for (int d = 0; d < space_dim; ++d) {
                        h_el_nodes(c, n, d) = x[d];
                    }
                }
            }
            Kokkos::deep_copy(ws.el_coords, h_el_nodes);
            worksets.push_back(ws);
        }
    }
    return worksets;
}

} // end namespace accopiatti
