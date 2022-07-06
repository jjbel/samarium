# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(BUILD_BENCHMARKS)
    find_package(benchmark 1.6 REQUIRED)
    add_executable(samarium_benchmarks main.cpp particles.cpp)

    target_link_libraries(samarium_benchmarks PUBLIC samarium::samarium)
    target_link_libraries(samarium_benchmarks PUBLIC benchmark::benchmark)

    include(find_deps)
    link_deps(samarium_benchmarks)
endif()
