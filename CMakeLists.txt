cmake_minimum_required(VERSION 3.8)
project(YUVLoader)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(TensorflowCC COMPONENTS Shared)

set(SOURCE_FILES
    src/main.cpp
    src/io/YUV.h
    src/io/YUV.cpp
    src/tf/Model.cpp
    src/tf/Model.h)

add_executable(YUVLoader ${SOURCE_FILES})

if(TensorflowCC_Shared_FOUND)
    target_link_libraries(YUVLoader LINK_PUBLIC TensorflowCC::Shared ${OpenCV_LIBS})
else()
    find_package(TensorflowCC REQUIRED COMPONENTS Static)
    target_link_libraries(YUVLoader LINK_PUBLIC TensorflowCC::Static ${OpenCV_LIBS})
endif()
