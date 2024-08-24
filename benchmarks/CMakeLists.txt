# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
    file(GLOB_RECURSE files "*.cpp")

    foreach(file ${files})
        get_filename_component(target_file_name ${file} NAME_WE)
        set(target_name "benchmark_${target_file_name}")
        message("Adding benchmark: ${target_name}")
        add_executable(${target_name} ${file})
        set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${target_file_name})
        target_link_libraries(${target_name} PUBLIC samarium::samarium)
        target_link_libraries(${target_name} PUBLIC benchmark::benchmark_main)
        link_deps(${target_name})
    endforeach()
endif()
