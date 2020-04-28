if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    # get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

cmake_minimum_required(VERSION 3.0)

option(CONFIG_CAFFE_SERIALIZER "caffe serializer" OFF)
option(CONFIG_ONNX_SERIALIZER "onnx serializer" OFF)
option(CONFIG_MXNET_SERIALIZER "mxnet serializer" OFF)
option(CONFIG_TF_SERIALIZER "tensorflow serializer" OFF)
option(CONFIG_TFLITE_SERIALIZER "tflite serializer" OFF)
option(CONFIG_DARKNET_SERIALIZER "darknet serializer" OFF)
option(CONFIG_SRC_TM_SERIALIZER "source tm serializer" OFF)
option(CONFIG_NO_OPT "do not use optimize flag for debugging" OFF)
option(CONFIG_ACL_OPENCL "build the acl opencl driver" OFF)
option(CONFIG_BUILD_TEST "build the test programs" OFF)
option(CONFIG_BUILD_CONVERT_TOOLS "build the test programs" OFF)
option(CONFIG_BUILD_SERIALIZER "build the test programs" OFF)

if(CONFIG_BUILD_CONVERT_TOOLS OR CONFIG_BUILD_SERIALIZER)
    set(CONFIG_CAFFE_SERIALIZER ON)
    set(CONFIG_ONNX_SERIALIZER ON)
    set(CONFIG_MXNET_SERIALIZER ON)
    set(CONFIG_TF_SERIALIZER ON)
    set(CONFIG_TFLITE_SERIALIZER ON)
    set(CONFIG_DARKNET_SERIALIZER ON)
    set(CONFIG_SRC_TM_SERIALIZER ON)
    set(CONFIG_NCNN_SERIALIZER ON)
endif()

if(OPEN_ACL_OPENCL)
    set(CONFIG_ACL_OPENCL ON)
endif()

if(CONFIG_BUILD_CONVERT_TOOLS OR CONFIG_BUILD_SERIALIZER)
    if( PROTOBUF_LIB_PATH )
        set(PROTOBUF_LIB "-L${PROTOBUF_LIB_PATH} -lprotobuf")
        set(PROTOBUF_CFLAGS "-I${PROTOBUF_INCLUDE_PATH}")
    else()
        find_package(Protobuf 3.0.0)
        if(PROTOBUF_FOUND)
            set(PROTOBUF_LIB ${Protobuf_LIBRARIES})
            set(PROTOBUF_CFLAGS -I${Protobuf_INCLUDE_DIRS})
        else()
            message(FATAL_ERROR "can not find protobuf library")
        endif()
    endif()
endif()

if(TENGINE_IN_BIND)
    set(CONFIG_TENGINE_ROOT ${CMAKE_SOURCE_DIR})
endif()

if(NOT DEFINED CONFIG_TENGINE_ROOT)
    message(FATAL_ERROR "please set CONFIG_TENGINE_ROOT")
endif()
message(STATUS "TENGINE_ROOT : ${CONFIG_TENGINE_ROOT}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Bsymbolic -Bsymbolic-functions")

if (NOT CONFIG_NO_OPT)
    add_definitions(-O2)
endif()

add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-g)
add_definitions(-funroll-loops)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual")
set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)

#plugin, build with internal Tengine header files
add_subdirectory(plugin)

# test program to test the modules
# OpenCV is necessary to build for most examples
if(CONFIG_BUILD_TEST)
    add_subdirectory(tests)
endif()

add_subdirectory(tools)

if(ANDROID AND (NOT TENGINE_IN_BIND))
    ADD_CUSTOM_TARGET(BUILD_DONE ALL COMMAND "")
    add_custom_command(TARGET BUILD_DONE
        POST_BUILD
        COMMAND bash android_install.sh
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
         )
endif()
