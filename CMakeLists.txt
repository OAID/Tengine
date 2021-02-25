# cmake version main changes:
#
# 3.11: generator expression for Xcode and Visual Studio added
#       INTERFACE support added
# 3.12: OpenMP support on Mac OS added
#       CUDA as a language added
#       -j,--parallel in --build added
#       new FindPython module added
#       C++20 support added
# 3.13: symbolic links on Windows added
#       target_link_options and add_link_options added
#       target_link_directories added
#       STATIC_LIBRARY_OPTIONS added
#       CUDA 10 (Turing) detection added
# 3.14: several utilities added
#       BUILD_RPATH_USE_ORIGIN added
# 3.15: some command line improve
# 3.16: Objective C and Objective C++ languages support added
#       target_precompile_headers support added
#       RPath control added
# 3.17: FindCUDAToolkit added
#       CUDA_RUNTIME_LIBRARY static linking mode added
#       FindPython with Conda added


# Check CMake version
CMAKE_MINIMUM_REQUIRED (VERSION 3.10 FATAL_ERROR)

# Disable inplace builds to prevent source tree corruption.
IF (" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    MESSAGE (FATAL_ERROR "FATAL: Building inplace are not allowed. You should create a separate directory for Building.")
ENDIF ()

# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# Enable the languages which in use
ENABLE_LANGUAGE (C CXX)
ENABLE_LANGUAGE (ASM)

if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

# include cmake helper list
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/check.cmake")        # Do check list
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/system_check.cmake)   # system check

project(tengine-lite)

# some basic options
option(TENGINE_ENABLE_ALL_SYMBOL "enable all symbol" ON)
option(TENGINE_OPENMP "openmp support" ON)
option(TENGINE_BUILD_BENCHMARK "build benchmark" ON)
option(TENGINE_BUILD_EXAMPLES "build examples" ON)
option(TENGINE_BUILD_TESTS "build tests" OFF)
option(TENGINE_COVERAGE "build for coverage" OFF)
option(TENGINE_BUILD_CPP_API "build C++ API" OFF)
option(TENGINE_DEBUG_DATA "extract data for every layer" OFF)
option(TENGINE_DEBUG_TIME "print time information for every layer" OFF)
option(TENGINE_DEBUG_MEM_STAT "print memory status for library" OFF)
option(TENGINE_ARCH_X86_AVX "build avx2 for x86" ON)
option(TENGINE_ARCH_ARM_82 "build armv8.2 for arm" OFF)

# some hcl options
option(TENGINE_STANDALONE_HCL "build standalone hcl lib" OFF)
option(TENGINE_STANDALONE_HCL_AUTO_LOAD "auto load standalone hcl lib" ON)

# some plugin options
option(TENGINE_ENABLE_ACL 		"Build with Arm Compute Library(ACL) support" OFF)
option(TENGINE_ENABLE_VULKAN 		"Build with Vulkan GPU compute support" OFF)
option(TENGINE_ENABLE_TENSORRT 		"Build with nVIDIA TensorRT support" OFF)
option(TENGINE_ENABLE_CUDABACKEND 	"Build with nVIDIA cuda support" OFF)
option(TENGINE_ENABLE_TIM_VX 		"Build with VSI Tensor Interface Module for OpenVX support" OFF)
option(TENGINE_ENABLE_NNIE        	"Build with HiSilicon NNIE support" OFF)

# add_definitions(-DCONFIG_DISABLE_PARAM_ACCESS)
# add_definitions(-DCONFIG_INTERN_ALLOCATOR)
add_definitions(-DCONFIG_TENGINE_SERIALIZER)

# Debug macro information
if (TENGINE_DEBUG_MEM_STAT)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE CONFIG_MEM_STAT)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE CONFIG_MEM_STAT)
endif()
if (TENGINE_DEBUG_DATA)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE DEBUG_DATA)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE DEBUG_DATA)
endif()
if (TENGINE_DEBUG_TIME)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE DEBUG_TIME)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE DEBUG_TIME)
endif()
if (TENGINE_LITE_VERSION)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE TENGINE_LITE_VERSION=${TENGINE_LITE_VERSION})
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE TENGINE_LITE_VERSION=${TENGINE_LITE_VERSION})
endif()
if (TENGINE_VERSION_POSTFIX)
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CXX_PRIVATE TENGINE_VERSION_POSTFIX=${TENGINE_VERSION_POSTFIX})
    LIST (APPEND TENGINE_COMPILE_DEFINITION_CUDA_PRIVATE TENGINE_VERSION_POSTFIX=${TENGINE_VERSION_POSTFIX})
endif()


#list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -g)
#list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -O0)
list(APPEND TENGINE_COMPILE_OPTIONS_C_PRIVATE -fPIC)

#list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -g)
list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -O0)
list(APPEND TENGINE_COMPILE_OPTIONS_CXX_PRIVATE -fPIC)

#list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -g)
#list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -O0)
list(APPEND TENGINE_COMPILE_OPTIONS_CUDA_PRIVATE -Xcompiler=-fPIC)


# OHOS 3.0.0.80 patch for ASM language
if (OHOS AND NOT DEFINED CMAKE_ASM_COMPILER_TARGET)
    set(CMAKE_ASM_COMPILER_TARGET ${OHOS_LLVM})
endif()

if (${TENGINE_TARGET_PROCESSOR} MATCHES "ARM")
    if (TENGINE_TARGET_PROCESSOR_32Bit)
        if (OHOS)
            add_definitions(-march=armv7-a)
            add_definitions(-mfpu=neon)
            add_definitions(-mfloat-abi=softfp)
        elseif (NOT ANDROID)
            add_definitions(-march=armv7-a)
            add_definitions(-mfpu=neon)
            add_definitions(-mfp16-format=ieee)
            add_definitions(-mfpu=neon-fp16)
        endif()
    endif()

    if (TENGINE_TOOLCHIN_FLAG)
        add_definitions ("${TENGINE_TOOLCHIN_FLAG}")
    endif()
endif()


# local include files
include_directories(include/)
include_directories(src/op/)


# add OpenMP support
set (TENGINE_VERBOSE ON)
if (NOT TENGINE_FORCE_SKIP_OPENMP)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_library_openmp.cmake)

    TENGINE_CHECK_LIB_OPENMP(TENGINE_OMP_FOUND)
    if(TENGINE_OMP_FOUND AND TENGINE_VERBOSE)
        message (STATUS "TENGINE: OpenMP was found.")
    endif()
endif()


# enable cmake ctest module
enable_testing()

# src files
add_subdirectory(src)
if (TENGINE_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
if (TENGINE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if (TENGINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Report summary
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/summary.cmake")
