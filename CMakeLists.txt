cmake_minimum_required(VERSION 3.16)
project(samarium_dev LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# make sure that the file is being used as an entry point
include(modern_project_structure)
ensure_entry_point()

include(ccache)
use_ccache()

add_subdirectory(src)

# set restrictive compilation warnings
include(compiler_warnings)
set_project_warnings(samarium)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

option(SAMARIUM_EXAMPLE "Create an example executable from this file" OFF)
if(SAMARIUM_EXAMPLE)
    add_subdirectory(examples)
endif()

option(SAMARIUM_TESTS "Create tests for samarium" ON)
if(SAMARIUM_TESTS)
    include(CTest)
    add_subdirectory(test)
    enable_testing()

    add_test(NAME SamariumTestsMain COMMAND samarium_tests)

    include(valgrind)
    use_valgrind(SamariumMemTests ./test/bin/samarium_tests)
endif()

