cmake_minimum_required(VERSION 3.8)
project(YUVLoader)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES
    src/main.cpp
    src/io/YUV.h
    src/io/YUV.cpp)

add_executable(YUVLoader ${SOURCE_FILES})
target_link_libraries(YUVLoader ${OpenCV_LIBS})
