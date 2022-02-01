cmake_minimum_required(VERSION 3.16)
project(samarium CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(samarium
    samarium/Color.hpp
    samarium/Colors.hpp
    samarium/file.hpp
    samarium/Image.hpp
    samarium/math.hpp
    samarium/print.hpp
    samarium/Rect.hpp
    samarium/Renderer.hpp
    samarium/Renderer.cpp
    samarium/Shapes.hpp
    samarium/Transform.hpp
    samarium/util.hpp
    samarium/Vector2.hpp
)

target_link_libraries(samarium ${CONAN_LIBS})
