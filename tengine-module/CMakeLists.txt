if(CMAKE_TOOLCHAIN_FILE)
set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
# get absolute path, but get_filename_component ABSOLUTE only refer with source dir, so find_file here :(
get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

cmake_minimum_required(VERSION 3.0)

option(CONFIG_CAFFE_SERIALIZER "caffe serializer" ON)
option(CONFIG_ONNX_SERIALIZER "onnx serializer" ON)
option(CONFIG_MXNET_SERIALIZER "mxnet serializer" ON)
option(CONFIG_TF_SERIALIZER "tensorflow serializer" ON)
option(CONFIG_TFLITE_SERIALIZER "tflite serializer" ON)
option(CONFIG_DARKNET_SERIALIZER "darknet serializer" ON)
option(CONFIG_SRC_TM_SERIALIZER "source tm serializer" ON)
option(CONFIG_NO_OPT "do not use optimize flag for debugging" OFF)
option(CONFIG_ACL_OPENCL "build the acl opencl driver" OFF)
option(CONFIG_BUILD_TEST "build the test programs" OFF)
option(CONFIG_BUILD_CONVERT_TOOLS "build the test programs" OFF)

if( "${BUILD_SERIALIZER}" STREQUAL "y" )
	set( CONFIG_CAFFE_SERIALIZER ON )
	set( CONFIG_ONNX_SERIALIZER ON )
	set( CONFIG_MXNET_SERIALIZER ON )
	set( CONFIG_TF_SERIALIZER ON )
	set( CONFIG_TFLITE_SERIALIZER ON )
	set( CONFIG_DARKNET_SERIALIZER ON )
	set( CONFIG_SRC_TM_SERIALIZER ON )
elseif( "${BUILD_SERIALIZER}" STREQUAL "n" )
	set( CONFIG_CAFFE_SERIALIZER OFF )
	set( CONFIG_ONNX_SERIALIZER OFF )
	set( CONFIG_MXNET_SERIALIZER OFF )
	set( CONFIG_TF_SERIALIZER OFF )
	set( CONFIG_TFLITE_SERIALIZER OFF )
	set( CONFIG_DARKNET_SERIALIZER OFF )
	set( CONFIG_SRC_TM_SERIALIZER OFF )
endif()

if ( OPEN_ACL_OPENCL )
	set( CONFIG_ACL_OPENCL ON )
endif()

if (PROTOBUF_LIB_PATH)
set(PROTOBUF_LIB "-L${PROTOBUF_LIB_PATH} -lprotobuf")
else()
set(PROTOBUF_LIB  "-lprotobuf")
endif()
set(PROTOBUF_CFLAGS "-I${PROTOBUF_INCLUDE_PATH}")

if(NOT DEFINED CONFIG_TENGINE_ROOT)
    message(FATAL_ERROR "please set CONFIG_TENGINE_ROOT")
endif()
message(STATUS "TENGINE_ROOT : ${CONFIG_TENGINE_ROOT}")

set(CMAKE_INSTALL_PREFIX "${CONFIG_TENGINE_ROOT}/install")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Bsymbolic -Bsymbolic-functions")

if (NOT CONFIG_NO_OPT)
    add_definitions(-O2)
endif()

add_definitions(-Wall)
add_definitions(-fPIC)
add_definitions(-g)
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

