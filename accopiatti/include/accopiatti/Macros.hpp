#pragma once
#include <yaml-cpp/yaml.h>

#define INPUT_FILE_KEY_CHECK(SUBBLOCK, INPUT, KEY) \
    do { \
        if (!(INPUT[KEY])) { \
            throw std::runtime_error( \
                std::string("key \"") + KEY + \
                "\" required in " + SUBBLOCK + "!"); \
        } \
    } while (false)
