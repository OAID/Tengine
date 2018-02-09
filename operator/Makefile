CC=gcc  -std=gnu99
CXX=g++ -std=c++11
LD=g++


BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../include
INC_DIR+=-I$(shell pwd)/include 

MODULE_DIR=operator plugin
MODULE_NAME=liboperator.so

CXXFLAGS+=

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR)

libsubdir_list=$(MODULE_DIR)

PLUGIN_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

PLUGIN_SO:=$(BUILD_DIR)/$(MODULE_NAME)

TENGINE_SO=../build/core/libtengine.so

binsubdir_list=./tests

-include ./tests/Makefile

TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/tests/,$(bin-obj-y))
bin-obj-y:=


BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)


default: $(libsubdir_list) $(PLUGIN_SO)

$(PLUGIN_OBJS): $(libsubdir_list)

$(BIN_OBJS): $(binsubdir_list)

build: default install

test: $(BIN_EXES)

$(BIN_EXES):$(TENGINE_SO) $(PLUGIN_SO) $(BIN_OBJS)

$(PLUGIN_SO): $(PLUGIN_OBJS)
	@mkdir -p $(@D)
	$(CC) -o $@ -shared $^ $(LIBS)


$(libsubdir_list) $(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

install:
	@mkdir -p $(INSTALL_DIR)/lib $(INSTALL_DIR)/include
	@cp -af $(PLUGIN_SO) $(INSTALL_DIR)/lib

clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default install  test $(libsubdir_list) $(binsubdir_list)

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS) -ldl


SO_LIBS=-L ../build/operator -loperator -Wl,-rpath,./build/operator:./build/core -L ../build/core -ltengine

