
include_directories(serializer/include)

if(CONFIG_CAFFE_SERIALIZER)
    FILE(GLOB_RECURSE serializer_src "serializer/caffe/*.cpp")

    list(APPEND TENGINE_LIB_SRCS ${serializer_src})

    # the generated caffe.pb.cc

    set(caffe_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/serializer/caffe/te_caffe.pb.cc)

    set(caffe_dir ${CMAKE_CURRENT_SOURCE_DIR}/serializer/caffe)
    set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR}/serializer/caffe)

    ADD_CUSTOM_COMMAND(OUTPUT ${caffe_proto_cc}
                       COMMAND mkdir -p ${dest_dir}
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${caffe_dir} ${caffe_dir}/te_caffe.proto
                       COMMAND mv ${dest_dir}/te_caffe.pb.h ${caffe_dir}/../include/
                       DEPENDS ${caffe_dir}/te_caffe.proto)

    ADD_CUSTOM_TARGET(CAFFE_SERIALIZER_TARGET DEPENDS ${caffe_proto_cc})

    message("CAFFE SERIALIZER DEFINED: " ${caffe_proto_cc})
    list(APPEND TENGINE_LIB_SRCS ${caffe_proto_cc})
endif()

if(CONFIG_ONNX_SERIALIZER)
    FILE(GLOB_RECURSE serializer_src "serializer/onnx/*.cpp")

    list(APPEND TENGINE_LIB_SRCS ${serializer_src})

    # the generated pb.cc

    set(onnx_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/serializer/onnx/onnx.pb.cc)

    set(proto_dir ${CMAKE_CURRENT_SOURCE_DIR}/serializer/onnx)
    set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR}/serializer/onnx)

    ADD_CUSTOM_COMMAND(OUTPUT ${onnx_proto_cc}
                       COMMAND mkdir -p ${dest_dir}
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/onnx.proto
                       COMMAND mv ${dest_dir}/onnx.pb.h ${proto_dir}/../include/
                       DEPENDS ${proto_dir}/onnx.proto)

    ADD_CUSTOM_TARGET(ONNX_SERIALIZER_TARGET DEPENDS ${onnx_proto_cc})

    message("ONNX SERIALIZER DEFINED: " ${onnx_proto_cc})
    list(APPEND TENGINE_LIB_SRCS ${onnx_proto_cc})
endif()

if(CONFIG_TF_SERIALIZER)
    FILE(GLOB_RECURSE serializer_src "serializer/tensorflow/*.cpp")

    list(APPEND TENGINE_LIB_SRCS ${serializer_src})

    # the generated pb.cc

    set(tf_proto_cc ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/graph.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/function.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/node_def.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/op_def.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/attr_value.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/tensor.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/tensor_shape.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/types.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/versions.pb.cc
                    ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow/resource_handle.pb.cc)

    set(proto_dir ${CMAKE_CURRENT_SOURCE_DIR}/serializer/tensorflow)
    set(dest_dir  ${CMAKE_CURRENT_BINARY_DIR}/serializer/tensorflow)

    ADD_CUSTOM_COMMAND(OUTPUT ${tf_proto_cc}
                       COMMAND mkdir -p ${dest_dir}
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/graph.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/function.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/node_def.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/op_def.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/attr_value.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/tensor.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/tensor_shape.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/types.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/versions.proto
                       COMMAND protoc --cpp_out=${dest_dir} --proto_path=${proto_dir} ${proto_dir}/resource_handle.proto
                       COMMAND mv ${dest_dir}/*.pb.h ${proto_dir}/../include/)

    ADD_CUSTOM_TARGET(TF_SERIALIZER_TARGET DEPENDS ${tf_proto_cc})

    message("TF SERIALIZER DEFINED: " ${tf_proto_cc})
    list(APPEND TENGINE_LIB_SRCS ${tf_proto_cc})
endif()

if(CONFIG_TENGINE_SERIALIZER)
    include_directories(serializer/include/tengine)
    include_directories(serializer/include/tengine/v1)
    include_directories(serializer/include/tengine/v2)
    FILE(GLOB_RECURSE tengine_serializer_cpp_src "serializer/tengine/*.cpp")
    FILE(GLOB_RECURSE tengine_serializer_c_src "serializer/tengine/*.c")
    list(APPEND TENGINE_LIB_SRCS ${tengine_serializer_cpp_src} ${tengine_serializer_c_src})
endif()

if(CONFIG_MXNET_SERIALIZER)
    FILE(GLOB_RECURSE serializer_src "serializer/mxnet/*.cpp")
    list(APPEND TENGINE_LIB_SRCS ${serializer_src})
endif()

if(CONFIG_TFLITE_SERIALIZER)
    include_directories(serializer/include/tf_lite)
    FILE(GLOB_RECURSE tflite_serializer_src "serializer/tf_lite/*.cpp")
    list(APPEND TENGINE_LIB_SRCS ${tflite_serializer_src})
endif()


FILE(GLOB_RECURSE source_serializer_cpp_src "serializer/source/*.cpp")
list(APPEND TENGINE_LIB_SRCS ${source_serializer_cpp_src})

FILE(GLOB plugin_init "serializer/plugin/init.cpp")

list(APPEND TENGINE_LIB_SRCS ${plugin_init})
