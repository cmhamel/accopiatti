#pragma once
#include <accopiatti/FunctionSpace.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_DOFManager.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

template<Scalar T, int Dim>
class DofManager {
public:
    explicit DofManager(
        stk::ParallelMachine comm,
        Mesh& mesh,
        const YAML::Node& solution_inputs
    );

    auto get_basis(std::string block_name, std::string field_name) {
        return basis_map[block_name][field_name];
    }

    auto get_block_bases(std::string block_name) {
        return basis_map[block_name];
    }

    void get_field_offsets(std::vector<GlobalOrdinal>& indices, std::string block_name, int field_num) const {
        dof_manager->getGIDFieldOffsets(block_name, field_num);
    }

    void get_owned_and_ghosted_indices(std::vector<GlobalOrdinal>& indices) const {
        dof_manager->getOwnedAndGhostedIndices(indices);
    }

    void get_owned_indices(std::vector<GlobalOrdinal>& indices) const {
        dof_manager->getOwnedIndices(indices);
    }

    void summarize(std::ostream& os) const {
        dof_manager->printFieldInformation(os);
    }

private:
    void add_field(
        const int& spatial_dimension,
        const std::string& block,
        const std::string& field_name,
        stk::topology topology,
        const FunctionSpaceHelper& field
    );

    static std::map<std::string, FunctionSpaceHelper> parse_fields(const YAML::Node& solution_inputs) {
        std::map<std::string, FunctionSpaceHelper> fields;

        for (const auto& field : solution_inputs) {
            const auto name = field.first.as<std::string>();
            const auto config = field.second;

            fields.emplace(
                name,
                FunctionSpaceHelper{
                    config["basis order"].as<int>(),
                    parse_field_type(config["field type"].as<std::string>()),
                    parse_function_space(config["function space"].as<std::string>()),
                    config["integration order"].as<int>()
                }
            );
        }

        return fields;
    }

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

public:
    std::map<std::string, std::map<std::string, Teuchos::RCP<Basis<T>>>> basis_map;
    Teuchos::RCP<panzer::DOFManager> dof_manager;
};

} // end namespace accopiatti
