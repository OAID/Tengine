CC=${CROSS_COMPILE}gcc  -std=gnu99 ${SYSROOT_FLAGS}
CXX=${CROSS_COMPILE}g++ -std=c++11 ${SYSROOT_FLAGS}
LD=${CROSS_COMPILE}g++ ${SYSROOT_FLAGS}


BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/include  

MODULE_DIR=onnx  caffe plugin mxnet
MODULE_NAME=libserializer.so

CXXFLAGS+=

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR)
libsubdir_list=$(MODULE_DIR)

PLUGIN_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

PLUGIN_SO:=$(BUILD_DIR)/$(MODULE_NAME)

binsubdir_list=./tests

include ./tests/Makefile

TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/tests/,$(bin-obj-y))
bin-obj-y:=


BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)


default: $(libsubdir_list) $(PLUGIN_SO)

test: $(binsubdir_list) $(BIN_EXES)

$(PLUGIN_OBJS): $(libsubdir_list)
$(BIN_OBJS): $(binsubdir_list)



build: default install


install:
	@mkdir -p $(INSTALL_DIR)/lib
	@cp -f $(PLUGIN_SO) $(INSTALL_DIR)/lib

$(PLUGIN_SO): $(PLUGIN_OBJS)
	$(CC) -o $@ -shared $^ $(LIBS) $(PLUGIN_LDFLAGS)


$(libsubdir_list) $(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default install  $(libsubdir_list) $(binsubdir_list)

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS)

SO_LIBS=-L../build/core -Wl,-rpath,./build/core -ltengine


LIBS+=-lprotobuf -ldl

