# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

function(add_example file)
    get_filename_component(EXAMPLE_NAME ${file} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${file})
    link_deps(${EXAMPLE_NAME})
    target_link_libraries(${EXAMPLE_NAME} PUBLIC samarium)
endfunction(())

if(BUILD_EXAMPLES)
    # file(GLOB_RECURSE files "*.cpp")
    # foreach(file ${files})
    #     add_example(file)
    # endforeach()
    # TODO get all examples working
    add_example("./usage/input.cpp")
    add_example("./hello_world.cpp")
    add_example("./graphics/flow_field_noise.cpp")
endif()
