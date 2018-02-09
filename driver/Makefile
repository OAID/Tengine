CC=gcc  -std=gnu99
CXX=g++ -std=c++11
LD=g++


BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/include 
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../executor/include 

MODULE_DIR=rk3399  plugin
MODULE_NAME=libdriver.so

CXXFLAGS+=

COMMON_CFLAGS+= -Wall -g -fPIC  $(INC_DIR) 
ifeq ($(CONFIG_CAFFE_REF),y)
    COMMON_CFLAGS+= -DCONFIG_CAFFE_REF=1
endif

ifeq ($(CONFIG_ARCH_ARM64),y)
    COMMON_CFLAGS+= -DCONFIG_ARCH_ARM64=1
endif

libsubdir_list=$(MODULE_DIR)

PLUGIN_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

PLUGIN_SO:=$(BUILD_DIR)/$(MODULE_NAME)


binsubdir_list=

-include ./tests/Makefile

TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/tests/,$(bin-obj-y))
bin-obj-y:=


BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)


default: $(libsubdir_list) $(PLUGIN_SO)

TENGINE_SO=../build/core/libtengine.so

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
	$(LD) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS) $(LDFLAGS)


LIBS+=-lprotobuf -ldl 

SO_LIBS+=-Wl,-rpath,./build/operator:./build/serializer:./build/executor:./build/core
SO_LIBS+=-L ../build/executor -lexecutor -L../build/operator -loperator  -L ../build/serializer -lserializer
SO_LIBS+=-L ../build/core -ltengine -lopenblas

ifeq ($(CONFIG_CAFFE_REF),y)
    CAFFE_LIBS+=-Wl,-rpath,$(CAFFE_ROOT)/build/lib -L$(CAFFE_ROOT)/build/lib -lcaffe
    CAFFE_LIBS+= -lprotobuf -lboost_system -lglog
    PLUGIN_LDFLAGS+=$(CAFFE_LIBS)
endif

OPENCV_LDFLAGS=$(shell pkg-config --libs opencv)


LDFLAGS+=$(OPENCV_LDFLAGS)

