
cmake_minimum_required(VERSION 3.10)

project(UNET_TRT_PRO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CUDA_DIR     "/usr/local/cuda-11.7")
set(CUDNN_DIR    "/usr/local/cuda-11.7")

find_package(CUDA REQUIRED)

set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/opencv-3.4.10)
set(TensorRT_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/TensorRT-8.5.2.2)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_DIR}/include
    ${CUDA_DIR}/include
    ${TensorRT_DIR}/include
    ${CUDNN_DIR}/include
)

link_directories(
    ${OpenCV_DIR}/lib
    ${CUDA_DIR}/lib64
    ${TensorRT_DIR}/lib    
    ${CUDNN_DIR}/lib
)

file(GLOB_RECURSE SOURCE_LIST ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.h)

add_library(UNetLib SHARED ${SOURCE_LIST})
target_link_libraries(UNetLib cublas cudart opencv_world nvinfer nvonnxparser)

add_executable(change_detect_test main.cpp)
target_link_libraries(change_detect_test UNetLib)
