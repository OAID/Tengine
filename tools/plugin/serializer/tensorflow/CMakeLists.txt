
if(NOT DEFINED PROTOBUF_LIB)
    message(FATAL_ERROR "tensorflow serializer needs PROTOBUF_LIB, please  pass -DPROTOBUF_LIB=xxx when invoking cmake")
endif()

if(NOT DEFINED PROTOBUF_CFLAGS)
    message(FATAL_ERROR "tensorflow serializer needs PROTOBUF_CFLAGS, please  pass -DPROTOBUF_CFLAGS=xxx when invoking cmake")
endif()

set(SERIALIZER_TARGET tensorflow-serializer)
set(tensorflow_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR})
set(tensorflow_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/graph.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/function.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/node_def.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/op_def.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/attr_value.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/tensor.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/tensor_shape.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/types.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/versions.pb.cc
                        ${CMAKE_CURRENT_BINARY_DIR}/resource_handle.pb.cc)


FILE(GLOB_RECURSE TENSORFLOW_SRCS "*.cpp")

ADD_CUSTOM_COMMAND(OUTPUT ${tensorflow_proto_cc}
     COMMAND mkdir -p ${dest_dir}
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/graph.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/function.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/node_def.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/op_def.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/attr_value.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/tensor.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/tensor_shape.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/types.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/versions.proto
     COMMAND protoc --cpp_out=${dest_dir} --proto_path=${tensorflow_dir} ${tensorflow_dir}/resource_handle.proto)

ADD_LIBRARY(${SERIALIZER_TARGET} SHARED ${TENSORFLOW_SRCS} ${tensorflow_proto_cc})

ADD_CUSTOM_TARGET(tensorflow_proto_target DEPENDS ${tensorflow_proto_cc})

ADD_DEPENDENCIES(${SERIALIZER_TARGET} tensorflow_proto_target)

include_directories(${dest_dir})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROTOBUF_CFLAGS}")


target_link_libraries(${SERIALIZER_TARGET} ${TENGINE_LIBRARY} ${PROTOBUF_LIB})

SET_TARGET_PROPERTIES(${SERIALIZER_TARGET} PROPERTIES VERSION ${SERIALIZER_VERSION} SOVERSION ${SERIALIZER_SO_VERSION})

install(TARGETS ${SERIALIZER_TARGET} DESTINATION lib)

#below is for static library

ADD_LIBRARY(${SERIALIZER_TARGET}-static STATIC ${TENSORFLOW_SRCS} ${tensorflow_proto_cc})
SET_TARGET_PROPERTIES(${SERIALIZER_TARGET}-static PROPERTIES OUTPUT_NAME  ${SERIALIZER_TARGET})

ADD_DEPENDENCIES(${SERIALIZER_TARGET} ${SERIALIZER_TARGET}-static)

install(TARGETS ${SERIALIZER_TARGET}-static DESTINATION lib)
