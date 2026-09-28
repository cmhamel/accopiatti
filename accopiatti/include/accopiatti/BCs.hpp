#pragma once
#include <accopiatti/DofManager.hpp>
#include <accopiatti/Macros.hpp>
#include <accopiatti/Mesh.hpp>
#include "Panzer_BCStrategy_Factory_Defines.hpp"
#include "Panzer_Constant.hpp"
#include <Panzer_STK_SetupUtilities.hpp>
#include <Panzer_STK_Utilities.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

struct BCHelper {
    std::vector<std::string> components;
    std::string field;
    std::string function;
    std::string side_set;
    std::string type;
};

inline std::vector<size_t> get_sorted_unique_indices(const std::vector<GlobalOrdinal>& v) {
    // 1. Generate an initial index sequence [0, v.size() - 1]
    std::vector<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 2. Sort the indices based on the values in the original vector
    std::sort(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) {
        return v[i1] < v[i2];
    });

    // 3. Remove consecutive duplicate values using std::unique on the indices
    auto last = std::unique(indices.begin(), indices.end(), [&v](size_t i1, size_t i2) {
        return v[i1] == v[i2];
    });

    // 4. Shrink the indices vector to fit the unique elements
    indices.erase(last, indices.end());

    return indices;
}

template<Scalar T, int Dim>
auto get_sideset_gids(
    Mesh& mesh,
    DofManager<T, Dim>& dof_manager,
    const std::string& block_name,
    const std::string& sideset_name,
    const std::string& field_name
) {
    std::vector<stk::mesh::Entity> sides;
    mesh.mesh->getMySides(sideset_name, sides);
 
    std::vector<std::size_t> local_side_ids;
    std::vector<stk::mesh::Entity> elements;
    panzer_stk::workset_utils::getSideElements(
        *mesh.mesh, block_name,
        sides, local_side_ids, elements
    );

    constexpr int subcell_dim = Dim - 1;
    std::vector<int> basis_ids;
    std::vector<GlobalOrdinal> global_ids;
    for (std::size_t i = 0; i < elements.size(); ++i) {
        const LocalOrdinal local_elem_id = mesh.mesh->elementLocalId(elements[i]);
        const int side_ordinal = static_cast<int>(local_side_ids[i]);
 
        std::vector<GlobalOrdinal> elem_gids;
        dof_manager.get_element_global_dof_ids(block_name, local_elem_id, elem_gids);
 
        const auto& [offsets, basis_indices] = dof_manager.get_field_offsets_closure(
            block_name, field_name, subcell_dim, side_ordinal
        );

        // NOTE basis_indices above gives the index for the basis
        // associated with the gid accessed via offset.
        for (int off : offsets) {
            global_ids.push_back(elem_gids[off]);
        }

        for (int index : basis_indices) {
            basis_ids.push_back(index);
        }
    }

    return std::make_pair(basis_ids, global_ids);
}

struct DirichletBCs {
    std::vector<int> basis_ids;
    std::vector<int> bc_ids;
    std::vector<GlobalOrdinal> global_ids;
};

// TODO this needs to be heavily refined for different types of bcs
// and different potential function spaces for the fields.
// everything right now assumes Hgrad/Dirichlet
template<Scalar T, int Dim>
inline DirichletBCs setup_dirichlet_bcs(
    Mesh& mesh,
    DofManager<T, Dim>& dof_manager,
    std::map<std::string, FieldSetupHelper>& field_helpers,
    YAML::Node bc_inputs
) {
    if (!bc_inputs.IsSequence()) {
        std::cerr << "Error: Root node is not a sequence!" << std::endl;
        throw std::runtime_error("BC inputs needs to be a sequence!");
    }

    int bc_id = 0;
    std::vector<int> basis_ids;
    std::vector<int> bc_ids;
    std::vector<GlobalOrdinal> global_ids;
    for (const auto& item : bc_inputs) {
        // BCHelper bc;
        INPUT_FILE_KEY_CHECK("boundary conditions", item, "field");
        INPUT_FILE_KEY_CHECK("boundary conditions", item, "function");
        INPUT_FILE_KEY_CHECK("boundary conditions", item, "side sets");
        INPUT_FILE_KEY_CHECK("boundary conditions", item, "type");
        std::vector<std::string> components{};
        if (item["components"]) {
            components = item["components"].as<std::vector<std::string>>();
        }
    
        auto field = item["field"].as<std::string>();
        auto function = item["function"].as<std::string>();
        auto side_sets = item["side sets"].as<std::vector<std::string>>();
        auto type = item["type"].as<std::string>();

        if (type != "dirichlet") {
            continue;
        }

        std::vector<std::string> field_names;
        if (components.size() > 0) {
            for (const auto& component : components) {
                field_names.push_back(field + "_" + component);
            }
        } else {
            field_names.push_back(field);
        }

        // add checks to make sure field is in dof manager / field helpers
        const int& basis_order = field_helpers[field].basis_order;
        // std::vector<int> basis_ids;
        // std::vector<int> bc_ids;
        // std::vector<GlobalOrdinal> global_ids;
        for (const auto& side_set : side_sets) {
            auto side_set_blocks = mesh.get_sideset_element_block_names(side_set);
            for (const auto& block_name : side_set_blocks) {
                for (const auto& field_name : field_names) {
                    auto [temp_basis_ids, temp_global_ids] = get_sideset_gids<T, Dim>(mesh, dof_manager, block_name, side_set, field_name);
                    for (int i = 0; i < temp_basis_ids.size(); ++i) {
                        bc_ids.push_back(bc_id);
                    }
                    basis_ids.insert(basis_ids.end(), temp_basis_ids.begin(), temp_basis_ids.end());
                    global_ids.insert(global_ids.end(), temp_global_ids.begin(), temp_global_ids.end());
                }
            }
        }

        bc_id += 1;
    }
    assert(basis_ids.size() == bc_ids.size());
    assert(basis_ids.size() == global_ids.size());
    std::vector<size_t> indices = get_sorted_unique_indices(global_ids);
    std::vector<int> new_basis_ids;
    std::vector<int> new_bc_ids;
    std::vector<GlobalOrdinal> new_global_ids;
    for (const auto& index : indices) {
        new_basis_ids.push_back(basis_ids[index]);
        new_bc_ids.push_back(bc_ids[index]);
        new_global_ids.push_back(global_ids[index]);
    }

    std::cout << "Num global ids in bcs = " << new_global_ids.size();
    // DirichletBCs bc = DirichletBCs{basis_ids, bc_ids, global_ids};
    DirichletBCs bc = DirichletBCs{new_basis_ids, new_bc_ids, new_global_ids};
    return bc;
}

} // end namespace accopiatti
