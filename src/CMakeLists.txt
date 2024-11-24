# SPDX-License-Identifier: MIT Copyright (c) 2022-2024 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/jjbel/samarium>

cmake_minimum_required(VERSION 3.30)

if(SAMARIUM_ENABLE_CUDA)
    set(langs "CXX CUDA")
else()
    set(langs "CXX")
endif()

project(
    samarium
    VERSION 1.1.0
    DESCRIPTION "2-D physics simulation engine written in modern C++"
    HOMEPAGE_URL "https://github.com/jjbel/samarium"
    LANGUAGES "${langs}"
)

# why not target_compile_features(particles PUBLIC cxx_std_11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE SAMARIUM_SOURCES "*.*pp")

if(SAMARIUM_ENABLE_CUDA)
    # set(CMAKE_CUDA_FLAGS "-O3")
    # enable_language(CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "*.cu*")
    list(APPEND SAMARIUM_SOURCES "${CUDA_SOURCES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 27")
endif()

# CUDA ARCHITECTURE
# TODO with 52, works but lots of error in calcn
# with 50, seemingly less error?
# with 70, doesn't work
# https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html : see all, native
# see: nvcc --list-gpu-code and nvcc --list-gpu-arch
# https://www.techpowerup.com/gpu-specs/geforce-mx330.c3493
# also using multiple arch's below...so chosen at runtime?

add_library(samarium ${SAMARIUM_SOURCES})
add_library(samarium::samarium ALIAS samarium)

if(SAMARIUM_ENABLE_CUDA)
    # device query says my MX330 supports 6.1
    set_target_properties(samarium PROPERTIES CUDA_ARCHITECTURES "50;60;61;70")
    # set_target_properties(samarium PROPERTIES CUDA_ARCHITECTURES "61")
endif()

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

option(SAMARIUM_HEADER_ONLY "Inline some cpp files for better performance" FALSE)
message(STATUS "samarium: SAMARIUM_HEADER_ONLY set to ${SAMARIUM_HEADER_ONLY}")
if(SAMARIUM_HEADER_ONLY)
    target_compile_definitions(samarium PUBLIC "SAMARIUM_HEADER_ONLY")
endif()

if(USE_WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

if(MSVC)
    # TODO Not needed to ignore warning?
    # set(FPNG_OPTIONS "/w")
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

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/samarium" # source directory
    DESTINATION "include" # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.glsl" # select header files
)

function(link_deps target)
    foreach(
        DEPENDENCY
        fmt
        range-v3
        stb
        tl-expected
        tl-function-ref
        bshoshany-thread-pool
        unordered_dense
        svector
        glm
        glad
        glfw3
        Freetype
    )
        if(NOT ${DEPENDENCY}_FOUND)
            find_package(${DEPENDENCY} CONFIG REQUIRED)
        endif()
    endforeach()

    foreach(
        DEPENDENCY
        fmt::fmt
        range-v3::range-v3
        stb::stb
        tl::expected
        tl::function-ref
        bshoshany-thread-pool::bshoshany-thread-pool
        unordered_dense::unordered_dense
        svector::svector
        glad::glad
        glfw
        glm::glm
        freetype
    )
        message(STATUS "Linking ${DEPENDENCY}")
        target_link_libraries(${target} PUBLIC ${DEPENDENCY})
    endforeach()

    if(USE_WARNINGS)
        target_compile_options(${target} PUBLIC ${WARNINGS})
    endif()

    message(STATUS "samarium: linking deps for ${target}")
endfunction()

link_deps(samarium)
