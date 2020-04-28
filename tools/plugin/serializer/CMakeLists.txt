#include directories

include_directories(${CONFIG_TENGINE_ROOT}/include)
include_directories(${CONFIG_TENGINE_ROOT}/core/include)
include_directories(${CONFIG_TENGINE_ROOT}/operator/include)

if(TENGINE_IN_BIND)
set(TENGINE_LIBRARY tengine)
else()
set(TENGINE_LIBRARY ${CONFIG_TENGINE_ROOT}/build/libtengine.so)
endif()

#the overall version for all serializers
set(SERIALIZER_VERSION 1)

#the so version for versoin switch
set(SERIALIZER_SO_VERSION 1)

add_definitions(-DSERIALIZER_VERSION=${SERIALIZER_VERSION})

if(CONFIG_CAFFE_SERIALIZER)
    add_definitions(-DCONFIG_CAFFE_SERIALIZER=1)
    add_subdirectory(caffe)
endif()

if(CONFIG_ONNX_SERIALIZER)
    add_definitions(-DCONFIG_ONNX_SERIALIZER=1)
    add_subdirectory(onnx)
endif()

if(CONFIG_MXNET_SERIALIZER)
    add_definitions(-DCONFIG_MXNET_SERIALIZER=1)
    add_subdirectory(mxnet)
endif()

if(CONFIG_TF_SERIALIZER)
    add_definitions(-DCONFIG_TF_SERIALIZER=1)
    add_subdirectory(tensorflow)
endif()

if(CONFIG_TFLITE_SERIALIZER)
    add_definitions(-DCONFIG_TFLITE_SERIALIZER=1)
    add_subdirectory(tf_lite)
endif()

if(CONFIG_SRC_TM_SERIALIZER)
    add_definitions(-DCONFIG_SRC_TM_SERIALIZER=1)
    add_subdirectory(src_tm)
endif()

if(CONFIG_DARKNET_SERIALIZER)
    add_definitions(-DCONFIG_DARKNET_SERIALIZER=1)
    add_subdirectory(darknet)
endif()

if(CONFIG_NCNN_SERIALIZER)
    add_definitions(-DCONFIG_NCNN_SERIALIZER=1)
    add_subdirectory(ncnn)
endif()
