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

option(SKIP_EXTRA_TESTS "Only run the main program and not the tests" OFF)
if(SKIP_EXTRA_TESTS)
    set(TEST_SOURCES "samarium_tests.cpp")
else()
    set(TEST_SOURCES
        tests/Vector2.cpp
        tests/concepts.cpp
        tests/main.cpp
    )
endif()

message(STATUS "Test sources: ${TEST_SOURCES}")
add_executable(samarium_tests ${TEST_SOURCES})

target_include_directories(samarium_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(samarium_tests PRIVATE samarium)
target_link_libraries(samarium_tests PRIVATE fmt::fmt)
target_link_libraries(samarium_tests PRIVATE sfml-graphics)

if(WARNINGS)
    target_compile_options(samarium_tests PRIVATE ${WARNINGS})
endif()
