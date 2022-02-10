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

function(conan_install CONAN_BUILD_TYPE)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/conan.lock)
        include(conan)

        set(CMAKE_BUILD_TYPE ${CONAN_BUILD_TYPE})
        conan_cmake_autodetect(settings)

        message(STATUS "Installing Conan dependenices... (this may take a few minutes)")
        conan_cmake_install(PATH_OR_REFERENCE ${CMAKE_SOURCE_DIR}
                            BUILD missing
                            REMOTE conancenter
                            SETTINGS ${settings}
                            OUTPUT_QUIET)
    else()
        message(STATUS "Conan dependencies already installed")
    endif()
endfunction(conan_install)
