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
project(samarium_dev LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# make sure that the file is being used as an entry point
include(modern_project_structure)
ensure_entry_point()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

option(RUN_CONAN "Run Conan through CMake itself" ON)
if(RUN_CONAN)
    include(conan_install)
    conan_install()
endif()

include(ccache)
use_ccache()

add_subdirectory(src)

# set restrictive compilation warnings
include(compiler_warnings)
set_project_warnings()
message(STATUS "Compiler options: ${WARNINGS}")
add_compile_options(${WARNINGS})

option(SAMARIUM_EXAMPLE "Create an example executable from this file" OFF)
if(SAMARIUM_EXAMPLE)
    add_subdirectory(examples)
endif()

option(SAMARIUM_TESTS "Create tests for samarium" ON)
if(SAMARIUM_TESTS)
    include(CTest)
    add_subdirectory(test)
    enable_testing()

    add_test(NAME SamariumTests COMMAND samarium_tests)

    # include(valgrind)
    # use_valgrind(SamariumTestsMem ./test/bin/samarium_tests_mem)
endif()
