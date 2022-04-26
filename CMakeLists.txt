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

option(USE_UBSAN "Generate coverage reports" FALSE)
if(USE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

include(conan_install)
conan_install()

include(ccache)
option(USE_CCACHE "Use Ccache" TRUE)
if(USE_CCACHE)
    use_ccache()
endif()

# set compilation warnings and flags
include(compiler_options)
set_compiler_options()
# message(STATUS "Compiler options: ${OPTIONS}")
add_compile_options(${OPTIONS})

add_subdirectory(src)

include(CTest)
enable_testing()
add_subdirectory(test)
add_test(NAME SamariumTests COMMAND samarium_tests)

option(GENERATE_COVERAGE "Generate coverage reports" FALSE)
include(coverage)
generate_coverage(samarium_tests)

option(USE_IWYU "Use include what you use" FALSE)
if(USE_IWYU)
    find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
    set_property(TARGET samarium_tests PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()
