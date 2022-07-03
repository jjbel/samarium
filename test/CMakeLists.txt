# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

execute_process(
    COMMAND ${CONAN_EXE} install catch2/3.0.1@ -if ${CMAKE_BINARY_DIR} -b missing -g
            cmake_find_package
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_QUIET
)

find_package(Catch2 3 REQUIRED)

add_executable(samarium_tests)

option(SAMARIUM_BUILD_EXTRA_TESTS "Build extra tests" ON)
if(SAMARIUM_BUILD_EXTRA_TESTS)
    add_subdirectory(tests)
    target_sources(samarium_tests PUBLIC ${SAMARIUM_TESTS_SOURCES})
    target_link_libraries(samarium_tests PUBLIC Catch2::Catch2WithMain)

    
    catch_discover_tests(samarium_tests)
else()
    target_sources(samarium_tests PUBLIC dev.cpp)
endif()

target_link_libraries(samarium_tests PUBLIC samarium::samarium)

include(find_deps)
link_deps(samarium_tests)
