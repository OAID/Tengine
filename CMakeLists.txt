if(CMAKE_TOOLCHAIN_FILE)
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")


cmake_minimum_required(VERSION 3.6)

set(CMAKE_BUILD_TYPE debug)
# set(CMAKE_BUILD_TYPE relwithdebinfo)
# set(CMAKE_BUILD_TYPE release)


project(tengine_android)

option(CONFIG_ARCH_ARM64 "build arm64 version" OFF)
option(CONFIG_ARCH_BLAS  "build blas  version" OFF)
option(CONFIG_CAFFE_SERIALIZER "caffe serializer" ON)
option(CONFIG_ONNX_SERIALIZER "onnx serializer" OFF)
option(CONFIG_TF_SERIALIZER "tensorflow serializer" OFF)

execute_process( COMMAND git rev-parse HEAD 
                 OUTPUT_VARIABLE  git_commit_id)


STRING(STRIP ${git_commit_id} stripped_commit_id)
set(GIT_COMMIT_ID -DGIT_COMMIT_ID="0x${stripped_commit_id}")

if (CONFIG_ARCH_ARM64)
   add_definitions(-DCONFIG_ARCH_ARM64=1)
endif()

if(CONFIG_ARCH_BLAS)
   add_definitions(-DCONFIG_ARCH_BLAS=1)
endif()

if(CONFIG_CAFFE_SERIALIZER)
   add_definitions(-DCONFIG_CAFFE_SERIALIZER=1)
endif()


add_definitions(${GIT_COMMIT_ID})
add_definitions(-Wno-unused-command-line-argument)
add_definitions(-Wall)
add_definitions(-Werror)
add_definitions(-fPIC)
add_definitions(-g)
add_definitions(-O3)
add_definitions(-funroll-loops)
add_definitions(-Wno-overloaded-virtual)
add_definitions(-Wno-deprecated-register)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)


include_directories(include)

include(cmake/core.cmake)
include(cmake/operator.cmake)
include(cmake/serializer.cmake)
include(cmake/executor.cmake)
include(cmake/driver.cmake)

set(can_use_assembler TRUE)
enable_language(ASM-ATT)


#add_subdirectory(signature)


FOREACH(src ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS} )
#if the file does not exist it is a generated file

if( NOT EXISTS ${src} )
# message ( ${src} " is a generated file" )
#   SET_SOURCE_FILES_PROPERTIES ( ${src} PROPERTIES  GENERATED  1)
#   set_property(SOURCE ${src} PROPERTY LANGUAGE C)
endif()

ENDFOREACH()


ADD_LIBRARY(tengine SHARED ${TENGINE_LIB_SRCS} ${TENGINE_SIGN_SRCS})

ADD_DEPENDENCIES(tengine KERNEL_ASM_TARGET)

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

if( CONFIG_CAFFE_SERIALIZER)
    ADD_DEPENDENCIES(tengine CAFFE_SERIALIZER_TARGET)
endif()

#Add OpenBLAS
if(CONFIG_ARCH_BLAS)
   if( BLAS_DIR)
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
          TARGET_LINK_LIBRARIES(tengine ${BLAS_DIR}/arm32/lib/libopenblas.so)
     endif()
     if(ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
       set(PROTOBUF_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
     endif()
   else()
      message(FATAL_ERROR "need to set the blas path")
   endif()
endif()

install (TARGETS tengine  DESTINATION lib)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_c_api.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/cpu_device.h DESTINATION include)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/include/tengine_test_api.h DESTINATION include)

