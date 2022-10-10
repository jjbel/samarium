# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(BUILD_EXAMPLES)
    file(GLOB_RECURSE files "*.cpp")
    foreach(file ${files})
        get_filename_component(EXAMPLE_NAME ${file} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${file})
        link_deps(${EXAMPLE_NAME})
        target_link_libraries(${EXAMPLE_NAME} PUBLIC samarium)
    endforeach()
endif()
