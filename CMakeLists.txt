# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(RUN_CONAN "Run Conan from CMake itself to automatically get dependenices" ON)
include(conan_install)

project(samarium_dev LANGUAGES CXX)

option(BUILD_UNIT_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_DOCS "Build documentation on running CMake" OFF)
option(BUILD_DOCS_TARGET "Build documentation as a target" ON)

option(USE_WARNINGS "Set useful compiler flags and warnings" ON)
option(USE_CCACHE "Use CCache to speed up compilation" OFF)
option(USE_LINKER "Change the default linker" "")
option(USE_UBSAN "Use Undefined Behaviour Sanitizer" OFF)

include(check_project_structure)
include(ccache)
include(compiler_options)
include(docs)
include(CTest)
include(include_what_you_use)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(benchmarks)

iwyu(samarium)
