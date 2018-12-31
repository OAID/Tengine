INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include
INC_DIR+=-I$(shell pwd)/../include
INC_DIR+=-I$(shell pwd)/include

MODULE_DIR+=

ifeq ($(CONFIG_CAFFE_SERIALIZER),y)
    obj-y+=caffe/
    COMMON_CFLAGS+= -DCONFIG_CAFFE_SERIALIZER
endif
ifeq ($(CONFIG_MXNET_SERIALIZER),y)
    obj-y+=mxnet/
    COMMON_CFLAGS+= -DCONFIG_MXNET_SERIALIZER
endif
ifeq ($(CONFIG_ONNX_SERIALIZER),y)
    obj-y+=onnx/
    COMMON_CFLAGS+= -DCONFIG_ONNX_SERIALIZER
endif
ifeq ($(CONFIG_TF_SERIALIZER),y)
    obj-y+=tensorflow/
    COMMON_CFLAGS+= -DCONFIG_TF_SERIALIZER
endif
ifeq ($(CONFIG_TFLITE_SERIALIZER),y)
    obj-y+=tf_lite/
    INC_DIR+=-I$(shell pwd)/include/tf_lite
    COMMON_CFLAGS+= -DCONFIG_TFLITE_SERIALIZER
endif
ifeq ($(CONFIG_TENGINE_SERIALIZER),y)
    obj-y+=tengine/
    COMMON_CFLAGS+= -DCONFIG_TENGINE_SERIALIZER
endif

obj-y+=source/
obj-y+=plugin/

CXXFLAGS+=
COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) -Werror



install:



