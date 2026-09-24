cmake_minimum_required(VERSION 3.28)

project(accopiatti VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
find_package(Panzer REQUIRED)
find_package(STK REQUIRED)
find_package(Trilinos REQUIRED)
find_package(yaml-cpp REQUIRED)

# Header-only core library.
file(GLOB_RECURSE ACCOPIATTI_SOURCES
    CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/accopiatti/*.cpp
)
add_library(accopiatti_core ${ACCOPIATTI_SOURCES})
add_library(accopiatti::core ALIAS accopiatti_core)

target_include_directories(accopiatti_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Trilinos_INCLUDE_DIRS}
)

target_compile_features(accopiatti_core PUBLIC cxx_std_20)

target_link_libraries(accopiatti_core
    PUBLIC
        MPI::MPI_CXX
        ${Trilinos_LIBRARIES}
        yaml-cpp::yaml-cpp
)

# Executable that consumes the header-only core.
add_executable(accopiatti src/main.cpp)

target_link_libraries(accopiatti PRIVATE accopiatti::core)