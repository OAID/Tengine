BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/include

-include $(MAKEFILE_CONFIG)

ifeq ($(CONFIG_TF_SERIALIZER),y)
    MODULE_DIR+=tensorflow
endif


ifeq ($(CONFIG_CAFFE_SERIALIZER),y)
    MODULE_DIR+=caffe
endif


MODULE_NAME=libwrapper.so

CXXFLAGS+=

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) -Werror

libsubdir_list=$(MODULE_DIR)

PLUGIN_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

PLUGIN_SO:=$(BUILD_DIR)/$(MODULE_NAME)

default: $(libsubdir_list) $(PLUGIN_SO)

$(PLUGIN_OBJS): $(libsubdir_list)


build: default install


install:
	@mkdir -p $(INSTALL_DIR)/lib
	@cp -f $(PLUGIN_SO) $(INSTALL_DIR)/lib
	@mkdir -p $(INSTALL_DIR)/wrapper_include
	@cp -rf $(shell pwd)/include/* $(INSTALL_DIR)/wrapper_include

$(PLUGIN_SO): $(PLUGIN_OBJS)
	$(CC) -o $@ -shared $^ $(LIBS) $(PLUGIN_LDFLAGS)

$(libsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list)
	@rm -rf $(PLUGIN_SO)

test:


.PHONY:  build clean default install test $(libsubdir_list)

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS)

SO_LIBS=-L../build/core -L../build/wrapper 
SO_LIBS+=-Wl,-rpath,./build/core:./build/wrapper
SO_LIBS+=-ltengine -lwrapper 


LIBS+=-lprotobuf  -lglog

