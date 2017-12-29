obj-y+=caffe_serializer.o
obj-y+=caffe.pb.o

caffe_serializer.o: caffe.pb.o

caffe.pb.cc: caffe.proto
	$(PROTOC) --cpp_out . caffe.proto
	$(MV) caffe.pb.h ../include/

PROTOC=protoc
MV=mv

caffe.pb_CXXFLAGS=-I. 
caffe_serializer_CXXFLAGS+=-I.

