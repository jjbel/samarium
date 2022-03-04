# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

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
message(STATUS "Compiler options: ${WARNINGS}")
add_compile_options(${WARNINGS})

add_subdirectory(src)

include(CTest)
include(setup_catch2)
setup_catch2()
enable_testing()
add_subdirectory(test)
add_test(NAME SamariumTests COMMAND samarium_tests)
