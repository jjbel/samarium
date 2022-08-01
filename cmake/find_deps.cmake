# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT SFML_FOUND)
    find_package(SFML CONFIG REQUIRED)
endif()

if(NOT fmt_FOUND)
    find_package(fmt CONFIG REQUIRED)
endif()

if(NOT range-v3_FOUND)
    find_package(range-v3 CONFIG REQUIRED)
endif()

if(NOT stb_FOUND)
    find_package(stb CONFIG REQUIRED)
endif()

if(NOT tl-expected_FOUND)
    find_package(tl-expected CONFIG REQUIRED)
endif()

function(link_deps target)
    target_link_libraries(${target} PUBLIC fmt::fmt)
    target_link_libraries(${target} PUBLIC range-v3::range-v3)
    target_link_libraries(${target} PUBLIC sfml::sfml)
    target_link_libraries(${target} PUBLIC stb::stb)
    target_link_libraries(${target} PUBLIC tl::expected)

    if(USE_WARNINGS)
        target_compile_options(${target} PUBLIC ${WARNINGS})
    endif()

    message(STATUS "Linking deps for ${target}")
endfunction()
