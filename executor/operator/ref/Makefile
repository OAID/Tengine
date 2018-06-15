obj-$(CONFIG_CAFFE_REF)+=caffe_operator.o
obj-y+=demo_operator.o

CAFFE_INCS += -I$(CAFFE_ROOT)/include
CAFFE_INCS += -I$(CAFFE_ROOT)/build/src
caffe_operator_CXXFLAGS=$(CAFFE_INCS) -DCPU_ONLY=1 -Wno-sign-compare

