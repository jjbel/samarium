# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.16)
project(samarium_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(SFML COMPONENTS system window graphics REQUIRED)
find_package(fmt REQUIRED)

if(NOT TEST_FILE)
    set(TEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/samarium_tests.cpp")
endif()

message(STATUS "Using test file: ${TEST_FILE}")
add_executable(samarium_tests ${TEST_FILE})

target_include_directories(samarium_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(samarium_tests PRIVATE samarium)
target_link_libraries(samarium_tests PRIVATE fmt::fmt)
target_link_libraries(samarium_tests PRIVATE sfml-graphics)
target_link_libraries(samarium_tests PRIVATE Catch2::Catch2WithMain)
# target_link_libraries(samarium_tests PRIVATE gcov)

if(WARNINGS)
    target_compile_options(samarium_tests PRIVATE ${WARNINGS})
endif()

# target_compile_options(samarium_tests PRIVATE "--coverage")
