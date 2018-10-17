obj-y+=tf_serializer.o

obj-y+=graph.pb.o
obj-y+=function.pb.o
obj-y+=node_def.pb.o
obj-y+=op_def.pb.o
obj-y+=attr_value.pb.o
obj-y+=tensor.pb.o
obj-y+=tensor_shape.pb.o
obj-y+=types.pb.o
obj-y+=versions.pb.o
obj-y+=resource_handle.pb.o

tf_serializer.o:graph.pb.o 
				function.pb.o 
				node_def.pb.o 
				op_def.pb.o 
				attr_value.pb.o 
				tensor.pb.o 
				tensor_shape.pb.o 
				types.pb.o 
				versions.pb.o 
				resource_handle.pb.o

graph.pb.cc: graph.proto
	$(PROTOC) --cpp_out . graph.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/tensorflow

attr_value.pb.cc: attr_value.proto
	$(PROTOC) --cpp_out . attr_value.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/tensorflow

function.pb.cc: function.proto
	$(PROTOC) --cpp_out . function.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

node_def.pb.cc: node_def.proto
	$(PROTOC) --cpp_out . node_def.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

op_def.pb.cc: op_def.proto
	$(PROTOC) --cpp_out . op_def.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

resource_handle.pb.cc: resource_handle.proto
	$(PROTOC) --cpp_out . resource_handle.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

tensor.pb.cc: tensor.proto
	$(PROTOC) --cpp_out . tensor.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

tensor_shape.pb.cc: tensor_shape.proto
	$(PROTOC) --cpp_out . tensor_shape.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

types.pb.cc: types.proto
	$(PROTOC) --cpp_out . types.proto 
	$(MV) *.h ../include/

versions.pb.cc: versions.proto
	$(PROTOC) --cpp_out . versions.proto 
	@mkdir -p ../include/tensorflow
	$(MV) *.h ../include/

clean::
	rm -f *.pb.cc ../include/tensorflow/*.pb.h


PROTOC=protoc
MV=mv

CXXFLAGS=-I../include/tensorflow

