# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.16)

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
find_package(fmt REQUIRED)

set(SAMARIUM_HEADERS
    ./samarium/core/concepts.hpp
    ./samarium/core/DynArray.hpp
    ./samarium/core/ThreadPool.hpp
    ./samarium/core/types.hpp
    ./samarium/core/version.hpp
    ./samarium/gui/sfml.hpp
    ./samarium/gui/Mouse.hpp
    ./samarium/gui/Keyboard.hpp
    ./samarium/math/Extents.hpp
    ./samarium/math/interp.hpp
    ./samarium/math/math.hpp
    ./samarium/math/shapes.hpp
    ./samarium/math/Transform.hpp
    ./samarium/math/Vector2.hpp
    ./samarium/math/vector_math.hpp
    ./samarium/math/Dual.hpp
    ./samarium/math/BoundingBox.hpp
    ./samarium/math/complex.hpp
    ./samarium/graphics/Color.hpp
    ./samarium/graphics/Image.hpp
    ./samarium/graphics/colors.hpp
    ./samarium/graphics/Gradient.hpp
    ./samarium/graphics/gradients.hpp
    ./samarium/graphics/Trail.hpp
    ./samarium/physics/Particle.hpp
    ./samarium/physics/collision.hpp
    ./samarium/physics/Spring.hpp
    ./samarium/physics/Car.hpp
    ./samarium/physics/fluid/Fluid.hpp
    ./samarium/util/file.hpp
    ./samarium/util/print.hpp
    ./samarium/util/random.hpp
    ./samarium/util/util.hpp
    ./samarium/util/terminal_dims.hpp
    ./samarium/util/date.hpp
    ./samarium/util/ostream.hpp
    ./samarium/util/Stopwatch.hpp
    ./samarium/util/format.hpp
    ./samarium/core.hpp
    ./samarium/graphics.hpp
    ./samarium/gui.hpp
    ./samarium/math.hpp
    ./samarium/physics.hpp
    ./samarium/util.hpp
    ./samarium/samarium.hpp
)

set(SAMARIUM_SOURCES
    ./samarium/core/ThreadPool.cpp
    ./samarium/gui/Mouse.cpp
    ./samarium/gui/Keyboard.cpp
    ./samarium/graphics/Trail.cpp
    ./samarium/physics/Car.cpp
    ./samarium/physics/collision.cpp
    ./samarium/physics/Spring.cpp
    ./samarium/util/file.cpp
    ./samarium/util/util.cpp
    ./samarium/util/RandomGenerator.cpp
    ./samarium/util/ostream.cpp
    ./samarium/util/Stopwatch.cpp
    ./samarium/App.cpp
)

add_library(samarium "${SAMARIUM_SOURCES}")

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
)

target_include_directories(samarium PUBLIC "${CMAKE_CURRENT_LIST_DIR}")

if(WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

target_link_libraries(samarium PRIVATE fmt::fmt)
target_link_libraries(samarium PRIVATE sfml-graphics)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT WIN32)
    find_package(TBB REQUIRED)
    message(STATUS "Using TBB")
    target_link_libraries(samarium PUBLIC TBB::tbb)
endif()

install(
    TARGETS samarium
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/src/samarium" # source directory
    DESTINATION "include" # target directory
    FILES_MATCHING # install only matched files
    PATTERN "*.hpp" # select header files
)
