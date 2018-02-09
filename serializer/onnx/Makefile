obj-y+=onnx_serializer.o
obj-y+=onnx.pb.o

onnx.pb.o: onnx_serializer.o

onnx.pb.cc: onnx.proto
	$(PROTOC) --cpp_out . onnx.proto
	$(MV) onnx.pb.h ../include

onnx.pb_CXXFLAGS=-I.
onnx_serializer_CXXFLAGS=-I.
PROTOC=protoc
MV=mv
RM=rm

clean::
	@$(RM) -f ../include/onnx.pb.h onnx.pb.cc

