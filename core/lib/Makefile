obj-y+=data_type.o
obj-y+=exec_context.o
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
obj-y+=tengine_c_helper.o
obj-y+=tengine_version.o
obj-y+=tengine_errno.o
obj-y+=tengine_runtime_error.o
obj-y+=logger/

obj-$(CONFIG_LEGACY_API)+=tengine_c_compat.o

flags-$(STATIC_BUILD)+=-DSTATIC_BUILD

compiler_CXXFLAGS+=$(flags-y)

COMMON_CFLAGS+=-I../../executor/include -I ../../operator/include
tengine_config_CXXFLAGS+=-DGIT_COMMIT_ID=\"$(GIT_COMMIT_ID)\"

ifneq ($(CONFIG_VERSION_POSTFIX),)
   tengine_version_CXXFLAGS+=-DCONFIG_VERSION_POSTFIX=\"$(CONFIG_VERSION_POSTFIX)\"
endif
