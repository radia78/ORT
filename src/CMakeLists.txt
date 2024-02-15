cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(onnx_cpp)

# REQUIRED PACKAGES
find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

# Add the executable scripts
add_executable(onnx_cpp main.cpp person_segmentator/person_segmentator.cpp)

# include opencv headers
target_include_directories(onnx_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

# include libraries
target_link_libraries(onnx_cpp PRIVATE "${OpenCV_LIBS}" onnxruntime::onnxruntime)

set_property(TARGET onnx_cpp PROPERTY CXX_STANDARD 17)