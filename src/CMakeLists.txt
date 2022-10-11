# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)
project(
    samarium
    VERSION 1.0.2
    DESCRIPTION "2-D physics simulation engine written in modern C++"
    HOMEPAGE_URL "https://github.com/strangeQuark1041/samarium"
    LANGUAGES CXX
)

option(SAMARIUM_HEADER_ONLY "Use Samarium using only header files" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

set(SAMARIUM_SOURCES
    # samarium/app/App.cpp
    samarium/gl/Context.cpp
    samarium/gl/draw.cpp
    samarium/gl/gl.cpp
    samarium/gl/Shader.cpp
    samarium/gl/Texture.cpp
    samarium/graphics/Trail.cpp
    samarium/gui/Mouse.cpp
    samarium/gui/Window.cpp
    samarium/gui/keyboard.cpp
    samarium/physics/collision.cpp
    samarium/physics/Particle.cpp
    samarium/physics/ParticleSystem.cpp
    samarium/physics/RigidBody.cpp
    samarium/physics/Spring.cpp
    samarium/util/file.cpp
    samarium/util/fpng/fpng.cpp
    samarium/util/noise.cpp
    samarium/util/ostream.cpp
    samarium/util/RandomGenerator.cpp
    samarium/util/Stopwatch.cpp
    samarium/util/util.cpp
)

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

message(STATUS "SAMARIUM_HEADER_ONLY set to ${SAMARIUM_HEADER_ONLY}")
if(SAMARIUM_HEADER_ONLY)
    target_compile_definitions(samarium PUBLIC "SAMARIUM_HEADER_ONLY")
endif()

if(USE_WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

if(MSVC)
    set(FPNG_OPTIONS "/w")
else()
    set(FPNG_OPTIONS "-fno-strict-aliasing;-msse4.1;-mpclmul;-w")
endif()

set_source_files_properties(
    "samarium/util/fpng/fpng.cpp" PROPERTIES COMPILE_OPTIONS "${FPNG_OPTIONS}"
)

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

function(link_deps target)
    foreach(
        DEPENDENCY
        SFML
        fmt
        range-v3
        stb
        tl-expected
        tl-function-ref
        bshoshany-thread-pool
        unordered_dense
        glm
        glad
        glfw3
    )
        if(NOT ${DEPENDENCY}_FOUND)
            find_package(${DEPENDENCY} CONFIG REQUIRED)
        endif()
    endforeach()

    foreach(
        DEPENDENCY
        fmt::fmt
        range-v3::range-v3
        sfml::sfml
        stb::stb
        tl::expected
        tl::function-ref
        bshoshany-thread-pool::bshoshany-thread-pool
        unordered_dense::unordered_dense
        glad::glad
        glfw
        glm::glm
    )
        target_link_libraries(${target} PUBLIC ${DEPENDENCY})
    endforeach()

    if(USE_WARNINGS)
        target_compile_options(${target} PUBLIC ${WARNINGS})
    endif()

    message(STATUS "Linking deps for ${target}")
endfunction()

link_deps(samarium)
