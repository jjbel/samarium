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

project(
    samarium
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(SFML 2.5 COMPONENTS system window graphics audio REQUIRED)
find_package(fmt REQUIRED)

add_library(samarium
    ${CMAKE_CURRENT_SOURCE_DIR}/samarium/core/ThreadPool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/samarium/util/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/samarium/util/file.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/samarium/graphics/Renderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/samarium/gui/Window.cpp
)

target_compile_features(samarium PUBLIC cxx_std_20)

target_include_directories(samarium PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(samarium PUBLIC "${SFML_INCLUDE_DIRS}")

if(WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

target_link_libraries(samarium PRIVATE fmt::fmt)
target_link_libraries(samarium PRIVATE sfml-graphics sfml-audio)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(TBB REQUIRED)
    target_link_libraries(samarium PRIVATE TBB::tbb)
endif()
