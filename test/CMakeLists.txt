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
project(samarium_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

find_package(SFML 2.5 COMPONENTS system window graphics audio REQUIRED)
find_package(fmt REQUIRED)

if(NOT TEST_FILE)
    set(TEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/samarium_tests.cpp")
endif()

message(STATUS "Using test file: ${TEST_FILE}")
add_executable(samarium_tests ${TEST_FILE})

target_include_directories(samarium_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(samarium_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/samarium)

target_link_libraries(samarium_tests PRIVATE samarium)
target_link_libraries(samarium_tests PRIVATE fmt::fmt)
target_link_libraries(samarium_tests PRIVATE sfml-graphics sfml-audio)

if(WARNINGS)
    target_compile_options(samarium_tests PRIVATE ${WARNINGS})
endif()
