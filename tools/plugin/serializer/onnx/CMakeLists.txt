if(NOT DEFINED PROTOBUF_LIB)
    message(FATAL_ERROR "onnx serializer needs PROTOBUF_LIB, please  pass -DPROTOBUF_LIB=xxx when invoking cmake")
endif()

if(NOT DEFINED PROTOBUF_CFLAGS)
    message(FATAL_ERROR "onnx serializer needs PROTOBUF_CFLAGS, please  pass -DPROTOBUF_CFLAGS=xxx when invoking cmake")
endif()

set(SERIALIZER_TARGET onnx-serializer)
set(onnx_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)
set(onnx_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR})

FILE(GLOB_RECURSE ONNX_SRCS "*.cpp")

ADD_CUSTOM_COMMAND(OUTPUT ${onnx_proto_cc}
                   COMMAND mkdir -p ${dest_dir}
                   COMMAND protoc --cpp_out=${dest_dir} --proto_path=${onnx_dir} ${onnx_dir}/onnx.proto
                   DEPENDS ${onnx_dir}/onnx.proto)

ADD_LIBRARY(${SERIALIZER_TARGET} SHARED ${ONNX_SRCS} ${onnx_proto_cc})

ADD_CUSTOM_TARGET(onnx_proto_target DEPENDS ${onnx_proto_cc})

ADD_DEPENDENCIES(${SERIALIZER_TARGET} onnx_proto_target)

include_directories(${dest_dir})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROTOBUF_CFLAGS}")


target_link_libraries(${SERIALIZER_TARGET} ${TENGINE_LIBRARY} ${PROTOBUF_LIB})

SET_TARGET_PROPERTIES(${SERIALIZER_TARGET} PROPERTIES VERSION ${SERIALIZER_VERSION} SOVERSION ${SERIALIZER_SO_VERSION})

install(TARGETS ${SERIALIZER_TARGET} DESTINATION lib)

#below is for static library

ADD_LIBRARY(${SERIALIZER_TARGET}-static STATIC ${ONNX_SRCS} ${onnx_proto_cc})
SET_TARGET_PROPERTIES(${SERIALIZER_TARGET}-static PROPERTIES OUTPUT_NAME  ${SERIALIZER_TARGET})

ADD_DEPENDENCIES(${SERIALIZER_TARGET} ${SERIALIZER_TARGET}-static)

install(TARGETS ${SERIALIZER_TARGET}-static DESTINATION lib)
