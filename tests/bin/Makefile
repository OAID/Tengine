bin-obj-y+=bench_sqz.o
bin-obj-y+=bench_mobilenet.o
bin-obj-y+=test_mxnet_sqz.o
bin-obj-y+=test_mxnet_mobilenet.o
bin-obj-y+=test_onnx_sqz.o
bin-obj-y+=vgg16.o
bin-obj-y+=test_deploy.o
bin-obj-y+=demo.o

ifeq ($(CONFIG_TF_SERIALIZER),y)
bin-obj-y+=test_tf_mobilenet.o
bin-obj-y+=test_tf_inceptionv3.o
endif

ifeq ($(CONFIG_TENGINE_SERIALIZER),y)
bin-obj-y+=test_tm.o
endif
