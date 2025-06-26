cmake_minimum_required(VERSION 3.16)

# Set CMake policy to suppress FindCUDA deprecation warning
cmake_policy(SET CMP0146 NEW)

# Check for HIP availability early to set languages
find_program(HIP_HIPCC_EXECUTABLE
    NAMES hipcc
    PATHS /opt/rocm/bin /usr/bin /usr/local/bin
    NO_DEFAULT_PATH
)

if(HIP_HIPCC_EXECUTABLE)
    project(opennlp_gpu_native LANGUAGES CXX HIP)
else()
    project(opennlp_gpu_native LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add compile flags for better math library support
if(WIN32)
    # Windows-specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /D_USE_MATH_DEFINES")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    endif()
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE")
    endif()
endif()

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

# Try to find CUDA first using modern CMake
if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        set(GPU_FOUND TRUE)
        set(GPU_PLATFORM "CUDA")
        set(GPU_LIBRARIES CUDA::cudart)
        set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
        set(GPU_COMPILE_DEFINITIONS "USE_CUDA")
        
        # Enable CUDA language
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        
        # CUDA-specific settings
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_50")
        
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
        
        # Enable HIP language for .hip files (commented out - using C++ with HIP headers instead)
        # enable_language(HIP)
        
        # Set HIP architectures (adjust for your specific GPUs)
        # set(CMAKE_HIP_ARCHITECTURES "gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100")
        
        # ROCm/HIP source files - use .cpp files with HIP headers for better compatibility
        set(GPU_SOURCES
            rocm/kernels.cpp
            rocm/HipKernelLaunchers.cpp
            rocm/HipOperations.cpp
        )
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
        # CUDA library using modern CMake
        add_library(opennlp_gpu_kernels STATIC ${GPU_SOURCES})
        target_link_libraries(opennlp_gpu_kernels ${GPU_LIBRARIES})
        target_include_directories(opennlp_gpu_kernels PRIVATE ${GPU_INCLUDE_DIRS})
        target_compile_definitions(opennlp_gpu_kernels PRIVATE ${GPU_COMPILE_DEFINITIONS})
    elseif(GPU_PLATFORM STREQUAL "ROCM")
        # ROCm/HIP library - mark kernel files as HIP source files
        set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
        add_library(opennlp_gpu_kernels STATIC ${GPU_SOURCES})
        target_link_libraries(opennlp_gpu_kernels ${GPU_LIBRARIES})
        target_compile_definitions(opennlp_gpu_kernels PRIVATE ${GPU_COMPILE_DEFINITIONS})
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

# Platform-specific math library linking
if(WIN32)
    # Windows: math functions are in the standard library
    # No additional linking needed for basic math functions
    if(MSVC)
        target_link_libraries(opennlp_gpu msvcrt)
    endif()
else()
    # Unix/Linux: link math library explicitly
    target_link_libraries(opennlp_gpu m)
endif()

target_compile_definitions(opennlp_gpu PRIVATE ${GPU_COMPILE_DEFINITIONS})
target_include_directories(opennlp_gpu PRIVATE 
    ${JNI_INCLUDE_DIRS}
    ${GPU_INCLUDE_DIRS}
)

# Platform-specific output configuration
if(WIN32)
    # Windows: Set DLL properties
    set_target_properties(opennlp_gpu PROPERTIES
        OUTPUT_NAME "opennlp_gpu"
        SUFFIX ".dll"
        PREFIX ""
    )
    
    # Enable DLL exports
    target_compile_definitions(opennlp_gpu PRIVATE 
        -DBUILDING_DLL
        -D_WINDOWS
        -D_USRDLL
    )
else()
    # Unix/Linux: Set shared library properties
    set_target_properties(opennlp_gpu PROPERTIES
        OUTPUT_NAME "opennlp_gpu"
        VERSION 1.0.0
        SOVERSION 1
    )
endif()

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
