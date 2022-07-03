# SPDX-License-Identifier: MIT Copyright (c) 2022 Jai Bellare See
# <https://opensource.org/licenses/MIT/> or LICENSE.md Project homepage:
# <https://github.com/strangeQuark1041/samarium>

cmake_minimum_required(VERSION 3.15)
project(samarium_dev LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(modern_project_structure)
include(ubsan)
include(ccache)
include(compiler_options)
include(docs)
include(conan_install)
include(find_deps)
include(CTest)
include(Catch)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
