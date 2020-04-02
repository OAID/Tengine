<<<<<<< HEAD
if(CMAKE_TOOLCHAIN_FILE)
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
message(STATUS,"CMAKE_TOOLCHAIN_FILE_NAME = ${CMAKE_TOOLCHAIN_FILE_NAME}, ${CMAKE_TOOLCHAIN_FILE}, ${CMAKE_SOURCE_DIR}")
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(FATAL_ERROR "cmake file only used for Android build")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Installation Directory")
message(STATUS,"CMAKE_CURRENT_SRC_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 3.6)

set(CMAKE_BUILD_TYPE debug)
# set(CMAKE_BUILD_TYPE relwithdebinfo)
# set(CMAKE_BUILD_TYPE release)


#get the NDK_ROOT from android.toolchains.cmake

set(PARSED_ANDROID_NDK_REGEX "(.+)/build/cmake/android.toolchain.cmake")

string(REGEX REPLACE "${PARSED_ANDROID_NDK_REGEX}" "\\1"  PARSED_ANDROID_NDK ${CMAKE_TOOLCHAIN_FILE})

file(READ "${PARSED_ANDROID_NDK}/source.properties" TE_NDK_SOURCE_PROPERTIES)

set(TE_NDK_REVISION_REGEX
  "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)?")
if(NOT TE_NDK_SOURCE_PROPERTIES MATCHES "${TE_NDK_REVISION_REGEX}")
  message(FATAL_ERROR "Failed to parse Android NDK revision: ${ANDROID_NDK}/source.properties.\n${TE_NDK_SOURCE_PROPERTIES}")
endif()

set(PARSED_NDK_MAJOR "${CMAKE_MATCH_1}")
set(PARSED_NDK_MINOR "${CMAKE_MATCH_2}")


#ndk less than 15, CMAKE 3.6.3 works, while higher version may failed
if(PARSED_NDK_MAJOR LESS 15)
if(CMAKE_VERSION VERSION_GREATER 3.6.3)
    message(FATAL_ERROR  "please use cmake at most VERSION 3.6.3 for ndk " ${PARSED_NDK_MAJOR} "." ${PARSED_NDK_MINOR})
endif()
=======
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Installation Directory")
    message(STATUS,"CMAKE_CURRENT_SRC_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

cmake_minimum_required(VERSION 2.8)

# set(CMAKE_BUILD_TYPE debug)
# set(CMAKE_BUILD_TYPE relwithdebinfo)
set(CMAKE_BUILD_TYPE release)

#get the NDK_ROOT from android.toolchains.cmake

if(ANDROID)
    set(PARSED_ANDROID_NDK_REGEX "(.+)/build/cmake/android.toolchain.cmake")

    string(REGEX REPLACE "${PARSED_ANDROID_NDK_REGEX}" "\\1"  PARSED_ANDROID_NDK ${CMAKE_TOOLCHAIN_FILE})

    file(READ "${PARSED_ANDROID_NDK}/source.properties" TE_NDK_SOURCE_PROPERTIES)

    set(TE_NDK_REVISION_REGEX
  "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)?")
    if(NOT TE_NDK_SOURCE_PROPERTIES MATCHES "${TE_NDK_REVISION_REGEX}")
        message(FATAL_ERROR "Failed to parse Android NDK revision: ${ANDROID_NDK}/source.properties.\n${TE_NDK_SOURCE_PROPERTIES}")
    endif()

    set(PARSED_NDK_MAJOR "${CMAKE_MATCH_1}")
    set(PARSED_NDK_MINOR "${CMAKE_MATCH_2}")

    #ndk less than 15, CMAKE 3.6.3 works, while higher version may failed
    if(PARSED_NDK_MAJOR LESS 15)
        if(CMAKE_VERSION VERSION_GREATER 3.6.3)
            message(FATAL_ERROR  "please use cmake at most VERSION 3.6.3 for ndk " ${PARSED_NDK_MAJOR} "." ${PARSED_NDK_MINOR})
        endif()
    endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
endif()

#real  project logic starts from here

<<<<<<< HEAD
project(tengine_android)

option(CONFIG_ARCH_ARM64 "build arm64 version" OFF)
option(CONFIG_ARCH_ARM32 "build arm32 version" OFF)
option(CONFIG_ARCH_BLAS  "build blas  version" OFF)
option(CONFIG_ARCH_ARM8_2 "build float16 for arm8.2" OFF)
option(CONFIG_ACL_GPU  "build acl gpu  version" OFF)
option(CONFIG_CAFFE_SERIALIZER "caffe serializer" ON)
option(CONFIG_ONNX_SERIALIZER "onnx serializer" OFF)
option(CONFIG_MXNET_SERIALIZER "mxnet serializer" OFF)
option(CONFIG_TF_SERIALIZER "tensorflow serializer" OFF)
option(CONFIG_TFLITE_SERIALIZER "tflite serializer" OFF)
option(CONFIG_TENGINE_SERIALIZER "tengine serializer" ON)
option(CONFIG_KERNEL_FP32 "KERNEL FP32" ON)
if(ANDROID_NDK_MAJOR AND CONFIG_ARCH_ARM8_2)
    option(CONFIG_KERNEL_FP16 "KERNEL FP16" ON)
endif()
option(CONFIG_KERNEL_INT8 "KERNEL INT8" ON)
option(CONFIG_KERNEL_UINT8 "KERNEL UINT8" ON)
option(CONFIG_AUTH_DEVICE  "AUTH DEVICE" ON)

=======
project(tengine)
set(BUILD_ALL_IN_ONE 1)
add_definitions(-DALL_IN_STATIC_LIB=1)

option(CONFIG_ARCH_X86 "build x86 version" OFF)
option(CONFIG_ARCH_X86_AVX "build avx2 for x86" OFF)
option(CONFIG_ARCH_ARM64 "build arm64 version" OFF)
option(CONFIG_ARCH_ARM32 "build arm32 version" OFF)
option(CONFIG_ARCH_ARM8_2 "build float16 for arm8.2" OFF)
option(CONFIG_TENGINE_SERIALIZER "tengine serializer" ON)
option(CONFIG_ONLINE_REPORT "online report" OFF)
option(CONFIG_KERNEL_FP32 "KERNEL FP32" ON)
if(ANDROID_NDK_MAJOR AND (ANDROID_NDK_MAJOR GREATER 18 ) )
    option(CONFIG_KERNEL_FP16 "KERNEL FP16" OFF)
endif()
option(CONFIG_KERNEL_INT8 "KERNEL INT8" OFF)
option(CONFIG_KERNEL_UINT8 "KERNEL UINT8" OFF)
option(CONFIG_AUTH_DEVICE  "AUTH DEVICE" OFF)
option(CONFIG_AUTHENICATION "AUTH DEVICE" OFF)
option(CONFIG_HCL_RELEASE  "HCL RELEASE" OFF)
option(CONFIG_AUTHED_FAILED_NOT_WORK  "AUTHED FAIELD TENGINE DO NOT WORK" OFF)
option(CONFIG_ACL_OPENCL  "Build Tengine module" OFF)
option(CONFIG_BUILD_CONVERT_TOOLS  "Build Tools" OFF)
option(CONFIG_BUILD_SERIALIZER "build serializer" OFF)

message(STATUS "CONFIG_ARCH_X86 = ${CONFIG_ARCH_X86}")
message(STATUS "CONFIG_ARCH_X86_AVX = ${CONFIG_ARCH_X86_AVX}")
message(STATUS "CONFIG_ARCH_ARM64 = ${CONFIG_ARCH_ARM64}")
message(STATUS "CONFIG_ARCH_ARM32 = ${CONFIG_ARCH_ARM32}")
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

#in face, this is related with run-time env, since API LEVEL 22 binary can run on API LEVEL 23 platform
if(ANDROID_PLATFORM_LEVEL LESS 23)
    add_definitions(-DNO_CXA_DEMANGLE)
endif()

<<<<<<< HEAD
set(CONFIG_VERSION_POSTFIX github)

#message("list dir ${CMAKE_CURRENT_LIST_DIR}/.git")
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.git)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_LIST_DIR} rev-parse HEAD
                    OUTPUT_VARIABLE git_commit_id)
=======
if(NOT CONFIG_VERSION_POSTFIX)
set(CONFIG_VERSION_POSTFIX github)
endif()

#message("list dir ${CMAKE_CURRENT_LIST_DIR}/.git")
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.git)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_LIST_DIR} rev-parse HEAD OUTPUT_VARIABLE git_commit_id)
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    STRING(STRIP ${git_commit_id} stripped_commit_id)
else()
    set(stripped_commit_id "UNKNOWN")
endif()

set(GIT_COMMIT_ID -DGIT_COMMIT_ID="0x${stripped_commit_id}")

message("GIT COMMIT ID: " 0x${stripped_commit_id})

<<<<<<< HEAD
=======
if( NOT ANDROID )
    add_definitions(-DCONFIG_LEGACY_API=1)
endif()

if (CONFIG_ARCH_X86)
    add_definitions(-DCONFIG_ARCH_X86=1)
    if (CONFIG_ARCH_X86_AVX)
        add_definitions(-mfma -mf16c)
    endif()
endif()

>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
if (CONFIG_ARCH_ARM64)
    add_definitions(-DCONFIG_ARCH_ARM64=1)
endif()

if(CONFIG_ARCH_ARM32)
    add_definitions(-DCONFIG_ARCH_ARM32=1)
<<<<<<< HEAD
endif()

if(CONFIG_ARCH_BLAS)
    add_definitions(-DCONFIG_ARCH_BLAS=1)
=======
    if(NOT ANDROID)
        add_definitions(-march=armv7-a -mfpu=neon -mfp16-format=ieee -mfpu=neon-fp16)
        set(ARCH_TYPE,"Arm32")
    endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
endif()

if(CONFIG_ARCH_ARM8_2)
    add_definitions(-DCONFIG_ARCH_ARM8_2=1)
    add_definitions(-mcpu=cortex-a55)
endif()

<<<<<<< HEAD
if(CONFIG_ACL_GPU)
    add_definitions(-DCONFIG_ACL_GPU=1)
endif()

if(CONFIG_CAFFE_SERIALIZER)
    add_definitions(-DCONFIG_CAFFE_SERIALIZER=1)
endif()

if(CONFIG_ONNX_SERIALIZER)
    add_definitions(-DCONFIG_ONNX_SERIALIZER=1)
endif()

if(CONFIG_MXNET_SERIALIZER)
    add_definitions(-DCONFIG_MXNET_SERIALIZER=1)
endif()

if(CONFIG_TF_SERIALIZER)
    add_definitions(-DCONFIG_TF_SERIALIZER=1)
endif()

if(CONFIG_TENGINE_SERIALIZER)
    add_definitions(-DCONFIG_TENGINE_SERIALIZER=1)
endif()

if(CONFIG_KERNEL_FP32)
    add_definitions(-DCONFIG_KERNEL_FP32=1)
endif()

if(CONFIG_KERNEL_FP16)
    add_definitions(-DCONFIG_KERNEL_FP16=1)
=======
add_definitions(-DCONFIG_TENGINE_SERIALIZER=1)

add_definitions(-DCONFIG_KERNEL_FP32=1)

if(CONFIG_KERNEL_FP16)
    if( ANDROID_NDK_MAJOR AND ( ${ANDROID_NDK_MAJOR} GREATER 18 ) )
        add_definitions(-DCONFIG_KERNEL_FP16=1)
    endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
endif()

if(CONFIG_KERNEL_INT8)
    add_definitions(-DCONFIG_KERNEL_INT8=1)
endif()

if(CONFIG_KERNEL_UINT8)
    add_definitions(-DCONFIG_KERNEL_UINT8=1)
endif()

<<<<<<< HEAD
if (CONFIG_VERSION_POSTFIX)
    add_definitions(-DCONFIG_VERSION_POSTFIX="${CONFIG_VERSION_POSTFIX}")
endif()

add_definitions(${GIT_COMMIT_ID})
add_definitions(-Wno-unused-command-line-argument)
add_definitions(-Wall)
add_definitions(-Werror)
add_definitions(-fPIC)
add_definitions(-g)
add_definitions(-O3)
add_definitions(-funroll-loops)
add_definitions(-Wno-deprecated-register)
add_definitions(-Wno-overloaded-virtual)
=======
if(CONFIG_AUTH_DEVICE)
    add_definitions(-DCONFIG_AUTH_DEVICE=1)
endif()
if (CONFIG_VERSION_POSTFIX)
    add_definitions(-DCONFIG_VERSION_POSTFIX="${CONFIG_VERSION_POSTFIX}")
endif()
if (CONFIG_ONLINE_REPORT)
    add_definitions(-DCONFIG_ONLINE_REPORT=1)
    add_definitions(-DENABLE_ONLINE_REPORT=1)
endif()

if (CONFIG_HCL_RELEASE)
    add_definitions(-DCONFIG_HCL_RELEASE=1)
endif()

if (CONFIG_AUTHED_FAILED_NOT_WORK)
    add_definitions(-DCONFIG_AUTHENICATION_DISABLEWORK=1)
endif()

add_definitions(${GIT_COMMIT_ID})
add_definitions(-Wno-unused-command-line-argument)
add_definitions(-fPIC)
add_definitions(-funroll-loops)
if(ANDROID)
    add_definitions(-Wno-deprecated-register)
    add_definitions(-Wno-overloaded-virtual)
    add_definitions(-O3)
else()
    add_definitions(-Wno-packed-bitfield-compat)
    add_definitions(-O2)
endif()
add_definitions("-O2 ${CONFIG_OPT_CFLAGS}")
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Bsymbolic -Bsymbolic-functions")

include_directories(include)

include(cmake/core.cmake)
include(cmake/operator.cmake)
include(cmake/serializer.cmake)
include(cmake/executor.cmake)
include(cmake/driver.cmake)

set(can_use_assembler TRUE)
enable_language(ASM-ATT)

<<<<<<< HEAD

#add_subdirectory(signature)

=======
if(CONFIG_ACL_OPENCL OR CONFIG_BUILD_CONVERT_TOOLS OR CONFIG_BUILD_SERIALIZER)
    set(TENGINE_IN_BIND 1)
    add_subdirectory(tools)
endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

FOREACH(src ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS} )
#if the file does not exist it is a generated file

if( NOT EXISTS ${src} )
# message ( ${src} " is a generated file" )
#   SET_SOURCE_FILES_PROPERTIES ( ${src} PROPERTIES  GENERATED  1)
#   set_property(SOURCE ${src} PROPERTY LANGUAGE C)
endif()

ENDFOREACH()

<<<<<<< HEAD
ADD_LIBRARY(hclcpu SHARED ${TOPERATOR_LIB_SRCS})
ADD_LIBRARY(tengine SHARED ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS})

#executor
ADD_DEPENDENCIES(hclcpu KERNEL_ASM_TARGET)
TARGET_LINK_LIBRARIES(hclcpu tengine)

#target_compile_definitions(operator,"--allow-shlib-undefined")

if(PROTOBUF_DIR)
    if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set(PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
    endif()
    if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
       set(PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
    endif()
    include_directories(${PROTOBUF_DIR}/include)
    TARGET_LINK_LIBRARIES(tengine ${PROTOBUF_LIB})
else()
    message(FATAL_ERROR "need to set the protobuf path")
endif()

if(CONFIG_CAFFE_SERIALIZER)
    ADD_DEPENDENCIES(tengine CAFFE_SERIALIZER_TARGET)
endif()

if(CONFIG_ONNX_SERIALIZER)
    ADD_DEPENDENCIES(tengine ONNX_SERIALIZER_TARGET)
endif()

if(CONFIG_TF_SERIALIZER)
    ADD_DEPENDENCIES(tengine TF_SERIALIZER_TARGET)
endif()

#Add OpenBLAS
if(CONFIG_ARCH_BLAS)
   if( BLAS_DIR)
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
          TARGET_LINK_LIBRARIES(tengine ${BLAS_DIR}/arm32/lib/libopenblas.so)
     endif()   
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
          TARGET_LINK_LIBRARIES(tengine ${BLAS_DIR}/arm64/lib/libopenblas.so)
     endif()
   else() 
      message(FATAL_ERROR "need to set the blas path")
   endif()
endif()

#Add ACL
if(CONFIG_ACL_GPU)
   if(ACL_ROOT)
        include_directories(${ACL_ROOT})
        include_directories(${ACL_ROOT}/include)
        if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
             TARGET_LINK_LIBRARIES(tengine ${ACL_ROOT}/build_32/libarm_compute_core.so ${ACL_ROOT}/build_32/libarm_compute.so)
        endif()   
        if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
             TARGET_LINK_LIBRARIES(tengine ${ACL_ROOT}/build_64/libarm_compute_core.so ${ACL_ROOT}/build_64/libarm_compute.so)
        endif()
      else() 
         message(FATAL_ERROR "need to set the ACL path")
   endif()
endif()

install (TARGETS tengine  DESTINATION lib)
install (TARGETS hclcpu  DESTINATION lib)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_c_api.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/cpu_device.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_c_compat.h DESTINATION include)
=======
if(BUILD_ALL_IN_STATIC)
    ADD_LIBRARY(tengine STATIC ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS} ${TOPERATOR_LIB_SRCS})
elseif(BUILD_ALL_IN_ONE)
    ADD_LIBRARY(tengine SHARED ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS} ${TOPERATOR_LIB_SRCS})
else()
    ADD_LIBRARY(hclcpu SHARED ${TOPERATOR_LIB_SRCS})
    ADD_LIBRARY(tengine SHARED ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS})
endif()

if(CONFIG_BUILD_CONVERT_TOOLS)
    ADD_LIBRARY(tengine-static STATIC ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS})
    SET_TARGET_PROPERTIES(tengine-static  PROPERTIES COMPILE_FLAGS "-DBUILD_TOOLS=1" )
    SET_TARGET_PROPERTIES(tengine-static PROPERTIES OUTPUT_NAME tengine)
endif()

if( NOT ANDROID)
    TARGET_LINK_LIBRARIES(tengine -lpthread -ldl)
endif()

if(BUILD_ALL_IN_STATIC OR BUILD_ALL_IN_ONE)
    ADD_DEPENDENCIES(tengine KERNEL_ASM_TARGET)
else()
    ADD_DEPENDENCIES(hclcpu KERNEL_ASM_TARGET)
    TARGET_LINK_LIBRARIES(hclcpu tengine)
endif()

if(AUTH_LIB)
    TARGET_LINK_LIBRARIES(hclcpu ${AUTH_LIB}/libauthd.a)
endif()

add_subdirectory(benchmark)

enable_testing()
add_subdirectory(tests)

add_subdirectory(examples)

install (TARGETS tengine  DESTINATION lib)
if(NOT BUILD_ALL_IN_STATIC AND NOT BUILD_ALL_IN_ONE)
    install (TARGETS hclcpu  DESTINATION lib)
endif()
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_c_api.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/cpu_device.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_c_compat.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_operations.h DESTINATION include)
if(ANDROID)
    install (FILES ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ABI}/libc++_shared.so DESTINATION lib)
endif()
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

