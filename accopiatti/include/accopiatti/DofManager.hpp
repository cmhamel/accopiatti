#pragma once
#include <accopiatti/FunctionSpace.hpp>
#include <accopiatti/Mesh.hpp>
#include <accopiatti/Types.hpp>
#include "Panzer_DOFManager.hpp"
#include "Teuchos_RCP.hpp"
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class DofManager {
public:
    explicit DofManager(
        stk::ParallelMachine comm,
        Mesh& mesh,
        const YAML::Node& solution_inputs
    ) {
        int spatial_dimension = mesh.get_dimension();
        dof_manager = Teuchos::rcp(new panzer::DOFManager);

        dof_manager->setConnManager(mesh.get_conn_manager(), comm);
        const auto block_tops = mesh.get_element_block_topologies();
        auto fields = parse_fields(solution_inputs);
        for (const auto& [field_name, field] : fields) {
            for (const auto& [block, topology] : block_tops) {
                FunctionSpace<double, 3> fspace = FunctionSpace<double, 3>(mesh, block, field, 1024);
                add_field(spatial_dimension, block, field_name, topology, field);
            }
        }

        dof_manager->buildGlobalUnknowns();
    }

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
    ) {
        auto basis = BasisFactory<double>::create(
            topology,
            field.function_space,
            field.basis_order
        );
        basis_map[block].emplace(field_name, basis);

        auto pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));

        if (field.field_type == FieldType::Scalar) {
            dof_manager->addField(block, field_name, pattern);
        } else if (field.field_type == FieldType::Vector) {
            std::vector<std::string> exts{"_x", "_y", "_z"};
            for (int i = 0; i < spatial_dimension; ++i) {
                dof_manager->addField(block, field_name + exts[i], pattern);
            }
        } else {
            throw std::runtime_error("Unsupported field type");
        }
    }

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
    std::map<std::string, std::map<std::string, Teuchos::RCP<Basis<double>>>> basis_map;
    Teuchos::RCP<panzer::DOFManager> dof_manager;
};

} // end namespace accopiatti
