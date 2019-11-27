
BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../serializer/include
INC_DIR+=-I$(shell pwd)/../driver/include
INC_DIR+=-I$(shell pwd)/../executor/include


COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g  $(INC_DIR) -Werror

binsubdir_list=./bin

include ./bin/Makefile

export BUILD_CONVERT_TOOLS

TOOLSDIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/, $(bin-obj-y))
bin-obj-y:=
BIN_OBJS=$(TOOLSDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)

default: $(libsubdir_list) $(binsubdir_list) $(TEST_LIB) $(BIN_EXES)

tools: $(BIN_EXES)


$(BIN_OBJS): $(binsubdir_list);

$(TOOLS_LIB_OBJS): $(libsubdir_list)

$(TOOLS_LIB): $(TOOLS_LIB_OBJS)
	ar crs $(TOOLS_LIB) $(TOOLS_LIB_OBJS)

build: default install


install:
	@mkdir -p $(INSTALL_DIR)/tool
	@cp -f $(BIN_EXES) $(INSTALL_DIR)/tool


$(libsubdir_list) $(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default test install  $(libsubdir_list) $(binsubdir_list)

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o  $(TOOLS_LIB)
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS) -L$(shell pwd)/../build

LIBS+=-Wl,-rpath,$(shell pwd)/../build -ltengine -lhclcpu -Wl,-allow-shlib-undefined
LIBS+=$(PROTOBUF_LIB)

SO_LIBS+=-Wl,-Bdynamic -ldl -lpthread -lresolv 
