# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)
project(
    samarium
    VERSION 1.0.0
    DESCRIPTION "2-D physics simulation engine written in modern C++"
    HOMEPAGE_URL "https://github.com/strangeQuark1041/samarium"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(
    SFML 2.5
    COMPONENTS system window graphics
    REQUIRED
)
find_package(fmt 7 REQUIRED)
find_package(range-v3 0.11.0 REQUIRED)

add_subdirectory(samarium)

add_library(samarium ${SAMARIUM_SOURCES})
add_library(samarium::samarium ALIAS samarium)

set_target_properties(
    samarium
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN YES
               VERSION "${PROJECT_VERSION}"
               SOVERSION "${PROJECT_VERSION_MAJOR}"
               EXPORT_NAME samarium
               OUTPUT_NAME samarium
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED YES
               CXX_EXTENSIONS NO
               WINDOWS_EXPORT_ALL_SYMBOLS YES
)

target_include_directories(samarium PUBLIC "${CMAKE_CURRENT_LIST_DIR}")

if(USE_WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

target_link_libraries(samarium PUBLIC fmt::fmt)
target_link_libraries(samarium PUBLIC range-v3::range-v3)
target_link_libraries(samarium PUBLIC sfml-graphics)

install(
    TARGETS samarium
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/samarium" # source directory
    DESTINATION "include" # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
)
