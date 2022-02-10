cmake_minimum_required(VERSION 3.16)
project(samarium CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_package(TBB REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(samarium "samarium/Renderer.cpp" "samarium/file.cpp")

target_link_libraries(samarium PRIVATE ${CONAN_LIBS})
target_link_libraries(samarium PRIVATE TBB::tbb)

if(WARNINGS)
    target_compile_options(samarium PUBLIC ${WARNINGS})
endif()

