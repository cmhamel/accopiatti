#pragma once
#include <stk_util/command_line/CommandLineParserParallel.hpp>

namespace accopiatti {

class CLIParser {
public:
    explicit CLIParser(stk::ParallelMachine comm);

    template<typename T>
    T get_option(std::string name) {
        return clp.get_option_value<T>(name);
    }

    std::pair<int, int> parse(int argc, char** argv);

private:
    stk::CommandLineParserParallel clp;
};

} // end namespace acoppiatti
