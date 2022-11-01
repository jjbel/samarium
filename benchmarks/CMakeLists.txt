# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(BUILD_BENCHMARKS)
    find_package(benchmark CONFIG REQUIRED)
    foreach(target random image_export particles)
        message("Adding benchmark: ${target}")
        add_executable(${target} "${target}.cpp")
        target_link_libraries(${target} PUBLIC samarium::samarium)
        target_link_libraries(${target} PUBLIC benchmark::benchmark_main)

        link_deps(${target})
    endforeach()
endif()
