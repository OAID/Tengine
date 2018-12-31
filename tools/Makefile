
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

CXXFLAGS+=

OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g  -fPIC  $(INC_DIR) $(OPENCV_CFLAGS) -Werror

binsubdir_list=./bin

include ./bin/Makefile

TOOLSDIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/,$(bin-obj-y))
bin-obj-y:=
BIN_OBJS=$(TOOLSDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)




default: $(libsubdir_list) $(binsubdir_list) $(TEST_LIB) $(BIN_EXES)

tools: $(BIN_EXES)


$(BIN_OBJS): $(binsubdir_list)

$(TOOLS_LIB_OBJS): $(libsubdir_list)

$(TOOLS_LIB): $(TOOLS_LIB_OBJS)
	ar crs $(TOOLS_LIB) $(TOOLS_LIB_OBJS)

build: default install


install:
	@mkdir -p $(INSTALL_DIR)/bin
	@cp -f $(BIN_EXES) $(INSTALL_DIR)/bin


$(libsubdir_list) $(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default test install  $(libsubdir_list) $(binsubdir_list)


$(BUILD_DIR)/%: $(BUILD_DIR)/%.o  $(TOOLS_LIB)
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(TENGINE_SO) $(LIBS) $(SO_LIBS)  -L$(shell pwd)/../build

#ugly workaround for cross compiling on arm64 ubuntu rootfs
OPENCV_LIB=$(shell pkg-config --libs-only-l --libs-only-L opencv) 

TENGINE_SO=-Wl,-rpath,$(shell pwd)/../build -ltengine -Wl,-allow-shlib-undefined
LIBS+=-lprotobuf -ldl -lpthread -lresolv
SO_LIBS+=$(OPENCV_LIB)


