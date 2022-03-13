# SPDX-License-Identifier: MIT
# Copyright (c) 2022 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.16)

project(
    samarium
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)
find_package(fmt REQUIRED)

add_library(samarium
    ./samarium/core/ThreadPool.cpp
    ./samarium/util/util.cpp
    ./samarium/util/file.cpp
    ./samarium/util/ostream.cpp
    ./samarium/graphics/Renderer.cpp
    ./samarium/gui/Window.cpp
    ./samarium/gui/Keymap.cpp
    ./samarium/graphics/Trail.cpp
)

target_compile_features(samarium PUBLIC cxx_std_20)

target_include_directories(samarium PUBLIC "${CMAKE_CURRENT_LIST_DIR}")

if(WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

target_link_libraries(samarium PUBLIC fmt::fmt)
target_link_libraries(samarium PUBLIC sfml-graphics)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT WIN32)
    find_package(TBB REQUIRED)
    message(STATUS "Using TBB")
    target_link_libraries(samarium PUBLIC TBB::tbb)
endif()
