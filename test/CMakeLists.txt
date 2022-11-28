# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

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
    add_test(NAME samarium_tests COMMAND samarium_tests)
    if(RUN_AFTER_BUILD)
        add_custom_target(
            run ALL
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/samarium_tests
            DEPENDS samarium_tests
            USES_TERMINAL
        )
    endif()
endif()

target_link_libraries(samarium_tests PUBLIC samarium::samarium)

link_deps(samarium_tests)
