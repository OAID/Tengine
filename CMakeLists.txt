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
endif()

#real  project logic starts from here

project(tengine)
set(BUILD_ALL_IN_ONE 1)
add_definitions(-DALL_IN_STATIC_LIB=1)

option(CONFIG_ARCH_X86 "build x86 version" OFF)
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
message(STATUS "CONFIG_ARCH_ARM64 = ${CONFIG_ARCH_ARM64}")
message(STATUS "CONFIG_ARCH_ARM32 = ${CONFIG_ARCH_ARM32}")

#in face, this is related with run-time env, since API LEVEL 22 binary can run on API LEVEL 23 platform
if(ANDROID_PLATFORM_LEVEL LESS 23)
    add_definitions(-DNO_CXA_DEMANGLE)
endif()

if(NOT CONFIG_VERSION_POSTFIX)
set(CONFIG_VERSION_POSTFIX github)
endif()

#message("list dir ${CMAKE_CURRENT_LIST_DIR}/.git")
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.git)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_LIST_DIR} rev-parse HEAD OUTPUT_VARIABLE git_commit_id)
    STRING(STRIP ${git_commit_id} stripped_commit_id)
else()
    set(stripped_commit_id "UNKNOWN")
endif()

set(GIT_COMMIT_ID -DGIT_COMMIT_ID="0x${stripped_commit_id}")

message("GIT COMMIT ID: " 0x${stripped_commit_id})

if( NOT ANDROID )
    add_definitions(-DCONFIG_LEGACY_API=1)
endif()

if (CONFIG_ARCH_X86)
    add_definitions(-DCONFIG_ARCH_X86=1)
endif()

if (CONFIG_ARCH_ARM64)
    add_definitions(-DCONFIG_ARCH_ARM64=1)
endif()

if(CONFIG_ARCH_ARM32)
    add_definitions(-DCONFIG_ARCH_ARM32=1)
    if(NOT ANDROID)
        add_definitions(-march=armv7-a -mfpu=neon -mfp16-format=ieee -mfpu=neon-fp16)
        set(ARCH_TYPE,"Arm32")
    endif()
endif()

if(CONFIG_ARCH_ARM8_2)
    add_definitions(-DCONFIG_ARCH_ARM8_2=1)
    add_definitions(-mcpu=cortex-a55)
endif()

add_definitions(-DCONFIG_TENGINE_SERIALIZER=1)

add_definitions(-DCONFIG_KERNEL_FP32=1)

if(CONFIG_KERNEL_FP16)
    if( ANDROID_NDK_MAJOR AND ( ${ANDROID_NDK_MAJOR} GREATER 18 ) )
        add_definitions(-DCONFIG_KERNEL_FP16=1)
    endif()
endif()

if(CONFIG_KERNEL_INT8)
    add_definitions(-DCONFIG_KERNEL_INT8=1)
endif()

if(CONFIG_KERNEL_UINT8)
    add_definitions(-DCONFIG_KERNEL_UINT8=1)
endif()

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

if(CONFIG_ACL_OPENCL OR CONFIG_BUILD_CONVERT_TOOLS OR CONFIG_BUILD_SERIALIZER)
    set(TENGINE_IN_BIND 1)
    add_subdirectory(tools)
endif()

FOREACH(src ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS} )
#if the file does not exist it is a generated file

if( NOT EXISTS ${src} )
# message ( ${src} " is a generated file" )
#   SET_SOURCE_FILES_PROPERTIES ( ${src} PROPERTIES  GENERATED  1)
#   set_property(SOURCE ${src} PROPERTY LANGUAGE C)
endif()

ENDFOREACH()

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

