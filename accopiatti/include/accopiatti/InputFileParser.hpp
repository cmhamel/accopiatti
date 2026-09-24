#pragma once
#include <filesystem>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class InputFileParser {
public:
    explicit InputFileParser(std::string file_name_);
        // : 
        // file_name(file_name_),
        // yaml_input(YAML::LoadFile(file_name)) {}

    YAML::Node get_input_block_raw(std::string key) {
        return yaml_input[key];
    }

    std::vector<std::string> get_input_block_keys(std::string key);
    // {
    //     std::vector<std::string> keys = {};
    //     if (yaml_input[key].IsMap()) {
    //         for (const auto& it : yaml_input[key]) {
    //             std::string temp_key = it.first.as<std::string>();
    //             keys.push_back(temp_key);
    //         }
    //     }
    //     return keys;
    // }

private:
    std::filesystem::path file_name;
    YAML::Node yaml_input;
};

}
