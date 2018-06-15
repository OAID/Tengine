obj-y+=net.o
obj-y+=blob.o
obj-y+=io.o
obj-y+=common.o
obj-y+=caffe.pb.o

caffe.pb.cc: caffe.proto
	$(PROTOC) --cpp_out . caffe.proto
	$(MV) caffe.pb.h ../include/

PROTOC=protoc
MV=mv
RM=rm

clean::
	@$(RM) -f ../include/caffe.pb.h caffe.pb.cc
