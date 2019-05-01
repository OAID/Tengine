BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

-include $(MAKEFILE_CONFIG)

INC_DIR+=-I$(shell pwd)/include 
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../serializer/include
INC_DIR+=-I$(shell pwd)/../driver/include
INC_DIR+=-I$(shell pwd)/../executor/include

CXXFLAGS+=

OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) $(OPENCV_CFLAGS) -Werror

binsubdir_list=./bin

include ./bin/Makefile

TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/,$(bin-obj-y))
bin-obj-y:=


BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)



default: $(binsubdir_list) $(BIN_EXES)


test: $(BIN_EXES)


$(BIN_OBJS): $(binsubdir_list) ;


build: default install


install:
	@mkdir -p $(INSTALL_DIR)/bin
	@cp -f $(BIN_EXES) $(INSTALL_DIR)/bin

$(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean:  $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default test install  $(binsubdir_list)


$(BUILD_DIR)/%: $(BUILD_DIR)/%.o 
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@  $(LIBS) $(SO_LIBS)


OPENCV_LIB=$(shell pkg-config --libs-only-l --libs-only-L opencv) 

LIBS+=-L ../build/ -ltengine -Wl,-allow-shlib-undefined

SO_LIBS+=-lprotobuf -ldl -lpthread
SO_LIBS+=-Wl,-rpath,./build/
SO_LIBS+=$(OPENCV_LIB)
