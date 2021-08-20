# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Copyright (c) 2021, OPEN AI LAB
# Author: xlchen@openailab.com
#

# add protobuf
find_package(Protobuf 3.6.1 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIR})

# ONNX
FILE(GLOB_RECURSE ONNX_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/onnx/*.cpp")
# generate pb.cc 
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx/onnx.proto)
list(APPEND ONNX_SERIALIZER_SRCS ${ONNX_PROTO_HDRS} ${ONNX_PROTO_SRCS})

# CAFFE
file(GLOB_RECURSE CAFFE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe/*.cpp")
# the generated caffe.pb.cc
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe/te_caffe.proto)
list(APPEND CAFFE_SERIALIZER_SRCS ${CAFFE_PROTO_HDRS} ${CAFFE_PROTO_SRCS})

# NCNN
file(GLOB_RECURSE NCNN_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/*.cpp")

# TENSORFLOW
file(GLOB_RECURSE TF_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/*.cpp")
# the generated pb.cc
protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HDRS 
    tensorflow/graph.proto 
    tensorflow/function.proto 
    tensorflow/node_def.proto 
    tensorflow/op_def.proto 
    tensorflow/attr_value.proto
    tensorflow/tensor.proto
    tensorflow/tensor_shape.proto
    tensorflow/types.proto
    tensorflow/versions.proto
    tensorflow/resource_handle.proto
)
list(APPEND TF_SERIALIZER_SRCS ${TF_PROTO_HDRS} ${TF_PROTO_SRCS})

# MXNET
file(GLOB_RECURSE MXNET_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mxnet/*.cpp")

# TFLITE
file(GLOB_RECURSE TFLITE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tflite/*.cpp" )
set(TFLITE_FLATBUFFERS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tflite)
include_directories(${TFLITE_FLATBUFFERS_PATH})

# SAVE GRAPH
FILE(GLOB_RECURSE SAVE_GRAPH_SRCS "${CMAKE_SOURCE_DIR}/tools/save_graph/*.cpp" "${CMAKE_SOURCE_DIR}/tools/save_graph/*.c")

# GRAPH OPTIMIZER
FILE(GLOB_RECURSE GRAPH_OPT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/utils/graph_optimizer/*.cpp")

# CONVERT TOOL
FILE(GLOB_RECURSE CONVERT_TOOL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/convert_tool.cpp")
list(APPEND CONVERT_TOOL_SRCS ${ONNX_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${CAFFE_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${NCNN_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${TF_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${MXNET_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${TFLITE_SERIALIZER_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${SAVE_GRAPH_SRCS})
list(APPEND CONVERT_TOOL_SRCS ${GRAPH_OPT_SRCS})


add_executable(convert_tool ${CONVERT_TOOL_SRCS})
IF(MSVC)
target_link_libraries(convert_tool ${CMAKE_PROJECT_NAME}-static)
ELSE()
target_link_libraries(convert_tool ${CMAKE_PROJECT_NAME}-static pthread dl m)
ENDIF()
target_link_libraries(convert_tool ${Protobuf_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_SOURCE_DIR}/source")
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_BINARY_DIR}/source")
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_SOURCE_DIR}/source/operator/prototype")
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_SOURCE_DIR}/tools")
TARGET_INCLUDE_DIRECTORIES (convert_tool PRIVATE "${CMAKE_SOURCE_DIR}/examples/common")
install(TARGETS convert_tool DESTINATION bin)

# add to a virtual project group
SET_PROPERTY(TARGET convert_tool PROPERTY FOLDER "tools/convert_tool")
