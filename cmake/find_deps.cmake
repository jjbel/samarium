# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

foreach(
    DEPENDENCY
    SFML
    fmt
    range-v3
    stb
    tl-expected
    tl-function-ref
    bshoshany-thread-pool
)
    if(NOT ${DEPENDENCY}_FOUND)
        find_package(${DEPENDENCY} CONFIG REQUIRED)
    endif()
endforeach()

function(link_deps target)
    foreach(
        DEPENDENCY
        fmt::fmt
        range-v3::range-v3
        sfml::sfml
        stb::stb
        tl::expected
        tl::function-ref
        bshoshany-thread-pool::bshoshany-thread-pool
    )
        target_link_libraries(${target} PUBLIC ${DEPENDENCY})
    endforeach()

    if(USE_WARNINGS)
        target_compile_options(${target} PUBLIC ${WARNINGS})
    endif()

    message(STATUS "Linking deps for ${target}")
endfunction()
