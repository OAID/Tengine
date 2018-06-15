obj-y+=caffe_serializer.o
obj-y+=te_caffe.pb.o

caffe_serializer.o: te_caffe.pb.o

te_caffe.pb.cc: te_caffe.proto
	$(PROTOC) --cpp_out . te_caffe.proto
	$(MV) te_caffe.pb.h ../include/

PROTOC=protoc
MV=mv
RM=rm

te_caffe.pb_CXXFLAGS=-I. 
caffe_serializer_CXXFLAGS+=-I.

clean::
	@$(RM) -f ../include/te_caffe.pb.h te_caffe.pb.cc
