cmake_minimum_required(VERSION 3.16)
project(opennlp_gpu_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# GPU Platform Detection and Configuration
option(USE_CUDA "Enable CUDA support" ON)
option(USE_ROCM "Enable ROCm/HIP support" ON)

set(GPU_FOUND FALSE)
set(GPU_LIBRARIES "")
set(GPU_INCLUDE_DIRS "")
set(GPU_COMPILE_DEFINITIONS "")
set(GPU_SOURCES "")

# Try to find CUDA first
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        set(GPU_FOUND TRUE)
        set(GPU_PLATFORM "CUDA")
        set(GPU_LIBRARIES ${CUDA_LIBRARIES})
        set(GPU_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        set(GPU_COMPILE_DEFINITIONS "USE_CUDA")
        
        # Enable CUDA language
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        
        # CUDA-specific settings
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_50)
        
        # CUDA source files
        set(GPU_SOURCES
            cuda/kernels.cu
            cuda/CudaKernelLaunchers.cu
            cuda/CudaOperations.cpp
        )
    endif()
endif()

# Try ROCm/HIP if CUDA not found or disabled
if(USE_ROCM AND NOT GPU_FOUND)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "ROCm/HIP found")
        set(GPU_FOUND TRUE)
        set(GPU_PLATFORM "ROCM")
        set(GPU_LIBRARIES hip::host)
        set(GPU_COMPILE_DEFINITIONS "USE_ROCM")
        
        # ROCm/HIP source files (will use .hip extension or .cpp with HIP headers)
        set(GPU_SOURCES
            rocm/kernels.hip
            rocm/HipKernelLaunchers.cpp
            rocm/HipOperations.cpp
        )
        
        # Alternative: if using .cpp files with HIP headers
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rocm/kernels.hip")
            set(GPU_SOURCES
                rocm/kernels.cpp
                rocm/HipKernelLaunchers.cpp
                rocm/HipOperations.cpp
            )
        endif()
    endif()
endif()

# Fallback to CPU-only if no GPU platform found
if(NOT GPU_FOUND)
    message(WARNING "Neither CUDA nor ROCm found. Building CPU-only version.")
    set(GPU_PLATFORM "CPU")
    set(GPU_COMPILE_DEFINITIONS "USE_CPU_ONLY")
    set(GPU_SOURCES
        cpu/CpuOperations.cpp
    )
endif()

message(STATUS "Building with GPU platform: ${GPU_PLATFORM}")

# Create GPU kernels library
if(GPU_FOUND)
    if(GPU_PLATFORM STREQUAL "CUDA")
        # CUDA library
        cuda_add_library(opennlp_gpu_kernels STATIC ${GPU_SOURCES})
        target_include_directories(opennlp_gpu_kernels PRIVATE ${GPU_INCLUDE_DIRS})
    elseif(GPU_PLATFORM STREQUAL "ROCM")
        # ROCm/HIP library
        add_library(opennlp_gpu_kernels STATIC ${GPU_SOURCES})
        target_link_libraries(opennlp_gpu_kernels ${GPU_LIBRARIES})
        
        # Set HIP compiler flags if using .hip files
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rocm/kernels.hip")
            set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
        endif()
    endif()
else()
    # CPU-only library
    add_library(opennlp_gpu_kernels STATIC ${GPU_SOURCES})
endif()

# Main JNI library
add_library(opennlp_gpu SHARED
    jni/GpuOperationsJNI.cpp
)

# Link libraries and set compile definitions
target_link_libraries(opennlp_gpu
    opennlp_gpu_kernels
    ${GPU_LIBRARIES}
)

target_compile_definitions(opennlp_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_include_directories(opennlp_gpu PRIVATE 
    ${JNI_INCLUDE_DIRS}
    ${GPU_INCLUDE_DIRS}
)

# Print configuration summary
message(STATUS "=== OpenNLP GPU Build Configuration ===")
message(STATUS "GPU Platform: ${GPU_PLATFORM}")
message(STATUS "GPU Libraries: ${GPU_LIBRARIES}")
message(STATUS "JNI Include Dirs: ${JNI_INCLUDE_DIRS}")
if(GPU_FOUND)
    message(STATUS "GPU Include Dirs: ${GPU_INCLUDE_DIRS}")
endif()
message(STATUS "Compile Definitions: ${GPU_COMPILE_DEFINITIONS}")
message(STATUS "========================================")

# Installation
install(TARGETS opennlp_gpu
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

# Optional: Create a config file for runtime GPU detection
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/gpu_config.h"
    @ONLY
)

target_include_directories(opennlp_gpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
