#pragma once
#include <yaml-cpp/yaml.h>

#define INPUT_FILE_KEY_CHECK(INPUT, KEY, SUBLOCK) \
    do { \
        if (!(INPUT[KEY])) { \
            throw std::runtime_error( \
                std::string("key \"") + KEY + \
                "\" required in " + SUBLOCK + "!"); \
        } \
    } while (false)
