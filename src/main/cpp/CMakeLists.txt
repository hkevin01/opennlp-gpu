cmake_minimum_required(VERSION 3.10)
project(opennlp_gpu_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA settings
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_50)

# Add CUDA kernels
cuda_add_library(opennlp_cuda_kernels STATIC
    cuda/kernels.cu
    cuda/CudaKernelLaunchers.cu
)

# Add JNI implementation
cuda_add_library(opennlp_cuda SHARED
    cuda/CudaOperations.cpp
)

# Link CUDA kernels to JNI implementation
target_link_libraries(opennlp_cuda
    opennlp_cuda_kernels
    ${CUDA_LIBRARIES}
)

# Installation
install(TARGETS opennlp_cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
