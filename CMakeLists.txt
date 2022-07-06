# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)
project(samarium_dev LANGUAGES CXX)

option(USE_WARNINGS "Set useful compiler flags and warnings" ON)
option(OPTIONS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(USE_CCACHE "Use CCache to speed up compilation" ON)
option(USE_UBSAN "Use Undefined Behaviour Sanitizer" OFF)

option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_COVERAGE "Generate coverage reports" OFF)
option(BUILD_DOCS "Build documentation" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(modern_project_structure)
include(ubsan)
include(ccache)
include(compiler_options)
include(docs)
include(conan_install)
include(find_deps)
include(CTest)
include(Catch)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)
include(coverage)
generate_coverage(samarium_tests)
