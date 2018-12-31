bin-obj-y+=bench_sqz.o
bin-obj-y+=bench_mobilenet.o
bin-obj-y+=test_mxnet_sqz.o
bin-obj-y+=test_mxnet_mobilenet.o
bin-obj-y+=test_onnx_sqz.o
bin-obj-y+=vgg16.o
bin-obj-y+=test_deploy.o
bin-obj-y+=demo.o
bin-obj-y+=test_perf_stat.o
bin-obj-y+=test_node_dump.o
bin-obj-y+=two_model_demo.o
bin-obj-y+=test_lstm.o

bin-obj-$(CONFIG_ACL_GPU)+=mt_mssd.o

ifeq ($(CONFIG_TF_SERIALIZER),y)
bin-obj-y+=test_tf_mobilenet.o
bin-obj-y+=test_tf_inceptionv3.o
bin-obj-y+=test_tf_resnet50.o
bin-obj-y+=test_tf.o
endif

ifeq ($(CONFIG_TENGINE_SERIALIZER),y)
bin-obj-y+=test_tm.o
bin-obj-y+=save_model_src.o
bin-obj-y+=load_model_src.o
endif

ifeq ($(CONFIG_ONNX_SERIALIZER),y)
bin-obj-y+=test_onnx.o
endif

ifeq ($(CONFIG_TFLITE_SERIALIZER),y)
bin-obj-y+=tf_lite_mssd.o
bin-obj-y+=tf_lite_mssd_quant.o
bin-obj-y+=tf_lite_mobilenet_quant.o
endif
