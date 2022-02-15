# MIT License

# Copyright (c) 2022

# Project homepage: <https://github.com/strangeQuark1041/samarium/>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

# For more information, please refer to <https://opensource.org/licenses/MIT/>

cmake_minimum_required(VERSION 3.16)
project(samarium CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES
    "samarium/core/concepts.hpp"
    "samarium/core/DynArray.hpp"
    "samarium/core/ThreadPool.hpp"
    "samarium/core/types.hpp"
    "samarium/core.hpp"

    "samarium/graphics/Color.hpp"
    "samarium/graphics/colors.hpp"
    "samarium/graphics/Image.hpp"
    "samarium/graphics/Renderer.hpp"
    "samarium/graphics/Renderer.cpp"
    "samarium/graphics.hpp"

    "samarium/gui/Window.hpp"
    "samarium/gui.hpp"

    "samarium/math/Extents.hpp"
    "samarium/math/interp.hpp"
    "samarium/math/math.hpp"
    "samarium/math/Rect.hpp"
    "samarium/math/shapes.hpp"
    "samarium/math/Transform.hpp"
    "samarium/math/Vector2.hpp"
    "samarium/math.hpp"

    "samarium/physics/Particle.hpp"
    "samarium/physics.hpp"

    "samarium/util/file.hpp"
    "samarium/util/file.cpp"
    "samarium/util/print.hpp"
    "samarium/util/random.hpp"
    "samarium/util/util.hpp"
    "samarium/util.hpp"

    "samarium/samarium.hpp"
)

add_library(samarium ${SOURCES})

target_include_directories(samarium PRIVATE "${CMAKE_SOURCE_DIR}/src/samarium")

target_link_libraries(samarium PRIVATE ${CONAN_LIBS})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(TBB REQUIRED)
    target_link_libraries(samarium PRIVATE TBB::tbb)
endif()

if(WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()
