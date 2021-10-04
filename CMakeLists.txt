cmake_minimum_required(VERSION 3.0)
project(samarium VERSION 1.0)

find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )

include_directories("/")
include_directories("include/extern")
include_directories("include/intern")

configure_file("include/intern/global_include/samariumConfig.h.in" samariumConfig.h)

add_executable(samarium samarium.cpp)
target_link_libraries(samarium ${OpenCV_LIBS})
