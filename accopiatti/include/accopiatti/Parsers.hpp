#pragma once
#include <filesystem>
#include <stk_util/command_line/CommandLineParserParallel.hpp>
#include <yaml-cpp/yaml.h>

namespace accopiatti {

class CLIParser {
public:
    explicit CLIParser(stk::ParallelMachine comm)
        : 
        clp(comm) {
        clp.add_flag("help,h", "display this help message and exit");
        // clp.add_required<int>(stk::CommandLineOption{"dimension", "d", "Dimension of problem"});
        clp.add_required<std::string>(stk::CommandLineOption{"input-file", "i", "Path to input file"});
        clp.add_optional<std::string>(stk::CommandLineOption{"log-file", "l", "Path to log file"}, "accopiatti.log");
    }

    template<typename T>
    T get_option(std::string name) {
        return clp.get_option_value<T>(name);
    }

    std::pair<int, int> parse(int argc, char** argv) {
        stk::CommandLineParser::ParseState p_state = clp.parse(argc, const_cast<const char**>(argv));

        if (p_state == stk::CommandLineParser::ParseError) {
            return std::make_pair(-1, 1);
        }

        if (p_state == stk::CommandLineParser::ParseHelpOnly) {
            std::cout << clp.get_usage() << std::endl;
            return std::make_pair(-1, 0);
        }

        return std::make_pair(0, 0);
    }

private:
    stk::CommandLineParserParallel clp;
};


class InputFileParser {
public:
    explicit InputFileParser(std::string file_name_)
        : 
        file_name(file_name_),
        yaml_input(YAML::LoadFile(file_name)) {

    }

    YAML::Node get_input_block_raw(std::string key) const {
        return yaml_input[key];
    }

    std::vector<std::string> get_input_block_keys(std::string key) const {
        std::vector<std::string> keys = {};
        if (yaml_input[key].IsMap()) {
            for (const auto& it : yaml_input[key]) {
                std::string temp_key = it.first.as<std::string>();
                keys.push_back(temp_key);
            }
        }
        return keys;
    }

private:
    std::filesystem::path file_name;
    YAML::Node yaml_input;
};

} // end namespace acoppiatti
