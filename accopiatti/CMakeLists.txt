# cmake_minimum_required(VERSION 3.28)
# project(accopiatti VERSION 1.0.0 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # deps
# find_package(Kokkos REQUIRED)
# find_package(MPI REQUIRED)
# find_package(Panzer REQUIRED)
# find_package(STK REQUIRED)
# find_package(Trilinos REQUIRED)
# find_package(yaml-cpp REQUIRED)

# # executables
# add_executable(accopiatti src/main.cpp)

# # compile options
# target_compile_options(accopiatti PRIVATE -w) # to suppress epetra deprecation warnings

# # include
# target_include_directories(accopiatti PRIVATE include)
# target_include_directories(accopiatti PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${Trilinos_INCLUDE_DIRS})

# # link
# target_link_libraries(accopiatti PRIVATE MPI::MPI_CXX)
# target_link_libraries(accopiatti PRIVATE ${Trilinos_LIBRARIES})
# target_link_libraries(accopiatti PRIVATE yaml-cpp::yaml-cpp)

# # sources
# # target_sources(accopiatti PRIVATE
# #     src/Mesh.cpp
# # )
cmake_minimum_required(VERSION 3.28)

project(accopiatti
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Let CMake invoke the compiler's module dependency scanner.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
find_package(Panzer REQUIRED)
find_package(STK REQUIRED)
find_package(Trilinos REQUIRED)
find_package(yaml-cpp REQUIRED)

# --------------------------------------------------------------------------
# Executable
# --------------------------------------------------------------------------

add_executable(accopiatti
    src/main.cpp

    # C++20 module interfaces
    # src/accopiatti/CLIParser.cppm
    # src/accopiatti/mesh.cppm
    # src/accopiatti/dof_manager.cppm
    # src/accopiatti/bcs.cppm
)

# --------------------------------------------------------------------------
# C++ modules
# --------------------------------------------------------------------------

# target_sources(accopiatti
#     PRIVATE
#         FILE_SET cxx_modules TYPE CXX_MODULES
#         FILES
#             src/accopiatti/cli.cppm
#             # src/accopiatti/mesh.cppm
#             # src/accopiatti/dof_manager.cppm
#             # src/accopiatti/bcs.cppm
# )
target_sources(accopiatti
    PRIVATE
        FILE_SET cxx_modules TYPE CXX_MODULES
        FILES
            src/accopiatti/CLIParser.cppm
)

# --------------------------------------------------------------------------
# Include paths
# --------------------------------------------------------------------------

target_include_directories(accopiatti
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Trilinos_INCLUDE_DIRS}
)

# --------------------------------------------------------------------------
# Compile options
# --------------------------------------------------------------------------

target_compile_options(accopiatti
    PRIVATE
        -w
)

# --------------------------------------------------------------------------
# Libraries
# --------------------------------------------------------------------------

target_link_libraries(accopiatti
    PRIVATE
        MPI::MPI_CXX
        ${Trilinos_LIBRARIES}
        yaml-cpp::yaml-cpp
)
