# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2024 Jai Bellare
# See <https://opensource.org/licenses/MIT/> or LICENSE.md
# Project homepage: <https://github.com/jjbel/samarium>

cmake_minimum_required(VERSION 3.15)

if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(
        FATAL_ERROR
            "samarium: '${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt' is meant to be used only "
            "as a CMake entry point and should not be included from other CMake files. "
            "Include '${CMAKE_CURRENT_SOURCE_DIR}/src/CMakeLists.txt' directly instead."
    )
endif()
