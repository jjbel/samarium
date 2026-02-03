# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/jjbel/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

add_executable(samarium_tests)

if(BUILD_UNIT_TESTS)
    find_package(Catch2 3 REQUIRED)
    file(GLOB_RECURSE SAMARIUM_TESTS_SOURCES "tests/*.cpp")
    target_sources(samarium_tests PUBLIC ${SAMARIUM_TESTS_SOURCES})
    target_link_libraries(samarium_tests PUBLIC Catch2::Catch2WithMain)

    include(Catch)
    catch_discover_tests(samarium_tests)
else()
    target_sources(samarium_tests PUBLIC dev.cpp)
    target_sources(samarium_tests PUBLIC dev.cpp)
    add_test(NAME samarium_tests COMMAND samarium_tests)
endif()

target_link_libraries(samarium_tests PUBLIC samarium::samarium)

link_deps(samarium_tests)
