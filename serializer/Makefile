INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include
INC_DIR+=-I$(shell pwd)/../include
INC_DIR+=-I$(shell pwd)/include

MODULE_DIR+=

ifeq ($(CONFIG_CAFFE_SUPPORT),y)
    obj-y+=caffe/
    COMMON_CFLAGS+= -DCONFIG_CAFFE_SERIALIZER
endif
ifeq ($(CONFIG_MXNET_SUPPORT),y)
    obj-y+=mxnet/
    COMMON_CFLAGS+= -DCONFIG_MXNET_SERIALIZER
endif
ifeq ($(CONFIG_ONNX_SUPPORT),y)
    obj-y+=onnx/
    COMMON_CFLAGS+= -DCONFIG_ONNX_SERIALIZER
endif
ifeq ($(CONFIG_TF_SUPPORT),y)
    obj-y+=tensorflow/
    COMMON_CFLAGS+= -DCONFIG_TF_SUPPORT
endif

obj-y+=plugin/

CXXFLAGS+=
COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) -Werror



install:



