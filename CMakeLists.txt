# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

cmake_minimum_required(VERSION 3.16)
project(samarium_dev LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(modern_project_structure)
ensure_entry_point()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(conan_install)
conan_install()

include(ccache)
use_ccache()

# set compilation warnings and flags
include(compiler_warnings)
set_project_warnings()
# message(STATUS "Compiler options: ${WARNINGS}")
add_compile_options(${WARNINGS})

add_subdirectory(src)

include(CTest)
enable_testing()
add_subdirectory(test)
add_test(NAME SamariumTests COMMAND samarium_tests)

option(GENERATE_COVERAGE "Generate coverage reports" FALSE)
include(coverage)
generate_coverage(samarium_tests)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
message(STATUS "HELLO?? ${CMAKE_EXPORT_COMPILE_COMMANDS} ??")
