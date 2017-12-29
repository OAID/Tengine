bin-obj-$(CONFIG_CAFFE_REF)+=test_caffe_executor.o

bin-obj-y+=test_buddy_executor.o
bin-obj-y+=test_onnx_executor.o
bin-obj-y+=test_image.o
bin-obj-y+=test_image2.o
bin-obj-y+=test_mobilenet.o
bin-obj-y+=test_googlenet.o
bin-obj-y+=test_graph_opt.o


ifeq ($(CONFIG_CAFFE_REF),y)
bin-obj-$(CONFIG_ARCH_ARM64)+=test_convolution.o
bin-obj-$(CONFIG_ARCH_ARM64)+=test_fc.o
bin-obj-$(CONFIG_ARCH_ARM64)+=test_pooling.o
#   bin-obj-$(CONFIG_ARCH_ARM64)+=test_gemm.o
bin-obj-$(CONFIG_ARCH_ARM64)+=test_scale.o
bin-obj-$(CONFIG_ARCH_ARM64)+=test_batch_norm.o
bin-obj-$(CONFIG_ARCH_ARM64)+=test_scale.o
endif

OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)

test_image_CXXFLAGS+=-I$(OPENCV_CFLAGS)
test_image_CXXFLAGS+=-I ../../serializer/include
test_image_CXXFLAGS+=-I ./include

test_caffe_executor_CXXFLAGS=-I ../../serializer/include
test_onnx_executor_CXXFLAGS=-I ../../serializer/include
test_buddy_executor_CXXFLAGS=-I ../../serializer/include

test_image2_CXXFLAGS=$(test_image_CXXFLAGS)
test_mobilenet_CXXFLAGS=$(test_image_CXXFLAGS)
test_googlenet_CXXFLAGS=$(test_image_CXXFLAGS)
test_graph_opt_CXXFLAGS=$(test_image_CXXFLAGS)

test_gemm_CXXFLAGS+=-I ../operator/arm/include
