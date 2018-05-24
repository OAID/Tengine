CXX=${CROSS_COMPILE}g++ -std=c++11 ${SYSROOT_FLAGS}
LD=${CROSS_COMPILE}g++ ${SYSROOT_FLAGS}


BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/include 
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../serializer/include
INC_DIR+=-I$(shell pwd)/../driver/include
INC_DIR+=-I$(shell pwd)/../executor/include

CXXFLAGS+=

OPENCV_CFLAGS=$(shell pkg-config --cflags opencv)

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR) $(OPENCV_CFLAGS)

libsubdir_list=./lib 

binsubdir_list=./bin

include ./bin/Makefile

TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/,$(bin-obj-y))
bin-obj-y:=



BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)


TEST_LIB_OBJS=$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))
TEST_LIB=$(BUILD_DIR)/libtest.a


TENGINE_SO=$(TOP_DIR)/build/core/libtengine.so
PLUGIN_SO=$(TOP_DIR)/build/operator/liboperator.so $(TOP_DIR)/build/serializer/libserializer.so $(TOP_DIR)/build/executor/libexecutor.so

default: $(libsubdir_list) $(binsubdir_list) $(TEST_LIB) $(BIN_EXES)

test: $(BIN_EXES)


$(BIN_OBJS): $(binsubdir_list)
$(TEST_LIB_OBJS): $(libsubdir_list)

$(TEST_LIB): $(TEST_LIB_OBJS)
	ar crs $(TEST_LIB) $(TEST_LIB_OBJS)

build: default install


install:
	@mkdir -p $(INSTALL_DIR)/bin
	@cp -f $(BIN_EXES) $(INSTALL_DIR)/bin


$(libsubdir_list) $(binsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(PLUGIN_SO) $(BIN_EXES)


.PHONY:  build clean default test install  $(libsubdir_list) $(binsubdir_list)


$(BUILD_DIR)/%: $(BUILD_DIR)/%.o  $(TEST_LIB)
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ -ltest $(TENGINE_SO) $(LIBS) $(SO_LIBS) -L ../build/tests/ 


OPENCV_LIB=$(shell pkg-config --libs opencv)

LIBS+=-lprotobuf -ldl -lpthread -rdynamic
SO_LIBS+=-L ../build/operator -L ../build/serializer -L ../build/executor -L ../build/driver
SO_LIBS+=-Wl,-rpath,./build/operator:./build/serializer:./build/executor:./build/driver${SYSROOT_RPATH}
SO_LIBS+=-ldriver -lexecutor -lserializer -loperator
SO_LIBS+=$(OPENCV_LIB)

