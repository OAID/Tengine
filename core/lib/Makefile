obj-y+=data_type.o
obj-y+=data_layout.o
obj-y+=tensor.o
obj-y+=tensor_shape.o
obj-y+=graph.o
obj-y+=node.o
obj-y+=static_graph.o
obj-y+=serializer.o
obj-y+=notify_instance.o
obj-y+=tengine_c_api.o
obj-y+=tengine_config.o
obj-y+=graph_executor.o
obj-y+=operator_manager.o
obj-y+=debug_utils.o
obj-y+=prof_record.o
obj-y+=tengine_plugin.o
obj-y+=compiler.o

obj-y+=logger/

flags-$(STATIC_BUILD)+=-DSTATIC_BUILD

compiler_CXXFLAGS+=$(flags-y)

#graph_executor_CXXFLAGS+=-I../../executor/include
tengine_config_CXXFLAGS+=-DGIT_COMMIT_ID=\"$(GIT_COMMIT_ID)\"


