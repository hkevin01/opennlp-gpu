cmake_minimum_required(VERSION 3.10)
project(opennlp_gpu_rocm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Find ROCm/HIP
if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to ROCm installation")
    endif()
endif()

set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH "Path to HIP installation")
include_directories(${ROCM_PATH}/include ${HIP_PATH}/include)
link_directories(${ROCM_PATH}/lib ${HIP_PATH}/lib)

# Source files
set(SOURCES
    RocmOperations.cpp
    RocmKernels.cpp
)

# Create shared library
add_library(opennlp_rocm SHARED ${SOURCES})
target_link_libraries(opennlp_rocm ${ROCM_PATH}/lib/libhip_runtime.so ${ROCM_PATH}/lib/libamdhip64.so)

# Installation
install(TARGETS opennlp_rocm
    LIBRARY DESTINATION lib
)
