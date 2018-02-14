CC=${CROSS_COMPILE}gcc  -std=gnu99 ${SYSROOT_FLAGS}
CXX=${CROSS_COMPILE}g++ -std=c++11 ${SYSROOT_FLAGS}
LD=${CROSS_COMPILE}g++ ${SYSROOT_FLAGS}



BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../serializer/include
INC_DIR+=-I$(shell pwd)/include

MODULE_DIR=caffe
MODULE_NAME=libwrapper.so

CXXFLAGS+=

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR)
libsubdir_list=$(MODULE_DIR)

PLUGIN_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

PLUGIN_SO:=$(BUILD_DIR)/$(MODULE_NAME)

default: $(libsubdir_list) $(PLUGIN_SO)

$(PLUGIN_OBJS): $(libsubdir_list)


build: default install


install:
	@mkdir -p $(INSTALL_DIR)/lib
	@cp -f $(PLUGIN_SO) $(INSTALL_DIR)/lib

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

SO_LIBS=-L../build/core -L../build/wrapper -L../build/serializer
SO_LIBS+=-Wl,-rpath,./build/core:./build/wrapper:./build/serializer
SO_LIBS+=-ltengine -lwrapper -lserializer

OPENCV_LIB=$(shell pkg-config --libs opencv)

#LIBS+=-lprotobuf -lboost_thread -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc -ldl
LIBS+=-lprotobuf -lboost_thread -lglog $(OPENCV_LIB)

