#include <accopiatti/InputFileParser.hpp>

namespace accopiatti {

InputFileParser::InputFileParser(std::string file_name_)
    : 
    file_name(file_name_),
    yaml_input(YAML::LoadFile(file_name)) {

}

std::vector<std::string> InputFileParser::get_input_block_keys(std::string key) {
    std::vector<std::string> keys = {};
    if (yaml_input[key].IsMap()) {
        for (const auto& it : yaml_input[key]) {
            std::string temp_key = it.first.as<std::string>();
            keys.push_back(temp_key);
        }
    }
    return keys;
}

} // end namespace accopiatti
