cmake_minimum_required(VERSION 3.16)
project(samarium_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(samarium_tests test.cpp)

set_target_properties(samarium_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_include_directories(samarium_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(samarium_tests samarium ${CONAN_LIBS})
