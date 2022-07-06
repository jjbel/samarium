# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(
    SFML 2.5
    COMPONENTS system window graphics
    REQUIRED
)
find_package(fmt 7 REQUIRED)
find_package(range-v3 0.11.0 REQUIRED)

function(link_deps target)
    target_link_libraries(${target} PUBLIC fmt::fmt)
    target_link_libraries(${target} PUBLIC range-v3::range-v3)
    target_link_libraries(${target} PUBLIC sfml-graphics)

    if(USE_WARNINGS)
        target_compile_options(${target} PUBLIC ${WARNINGS})
    endif()

    message(STATUS "Linking deps for ${target}")
endfunction()
