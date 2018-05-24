CC=${CROSS_COMPILE}gcc  -std=gnu99 ${SYSROOT_FLAGS}
CXX=${CROSS_COMPILE}g++ -std=c++11 ${SYSROOT_FLAGS}
LD=${CROSS_COMPILE}g++ ${SYSROOT_FLAGS}

BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS LD LDFLAGS CXXFLAGS COMMON_CFLAGS

INC_DIR+=-I$(shell pwd)/../include -I$(shell pwd)/include -I$(shell pwd)/include  
#INC_DIR+=-I$(shell pwd)/../operator/include


CXXFLAGS+= 

COMMON_CFLAGS+= -Wall -g -I$(shell pwd)/include -fPIC  $(INC_DIR)

libsubdir_list=lib

binsubdir_list=./tests 

-include tests/Makefile


TESTDIR_OBJS:=$(addprefix $(BUILD_DIR)/tests/,$(bin-obj-y))
bin-obj-y:=



-include bin/Makefile

BINDIR_OBJS:=$(addprefix $(BUILD_DIR)/bin/,$(bin-obj-y))
bin-obj-y:=

BIN_OBJS=$(TESTDIR_OBJS) $(BINDIR_OBJS)
BIN_EXES=$(BIN_OBJS:.o=)

TENGINE_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(libsubdir_list),$(f)/built-in.o))

TENGINE_LIB:=$(BUILD_DIR)/libtengine.a
TENGINE_SO:=$(BUILD_DIR)/libtengine.so

default: $(libsubdir_list) $(TENGINE_SO) 

test: $(binsubdir_list) $(BIN_EXES)

$(BIN_EXES): $(TENGINE_SO)  

$(TENGINE_OBJS): $(libsubdir_list)
$(BIN_OBJS): $(binsubdir_list)


build: default install

LDFLAGS+=-rdynamic ${SYSROOT_FLAGS}

#$(TENGINE_OBJS): $(libsubdir_list)
#$(BIN_OBJS):$(binsubdir_list)

$(BIN_EXES):$(BUILD_DIR)/%:$(BUILD_DIR)/%.o


$(TENGINE_SO): $(TENGINE_OBJS)
	$(CC) -o $@ -shared $(TENGINE_OBJS) $(LIBS)

$(TENGINE_LIB):$(TENGINE_OBJS)
	$(AR) crs $(TENGINE_LIB) $(TENGINE_OBJS)

$(binsubdir_list) $(libsubdir_list):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(libsubdir_list) $(binsubdir_list)
	@rm -rf $(TENGINE_SO) $(TENGINE_LIB) $(BIN_EXES)

install:
	@mkdir -p $(INSTALL_DIR)/lib $(INSTALL_DIR)/include
	@cp -f $(TENGINE_SO) $(INSTALL_DIR)/lib
	@cp -f include/tengine_c_api.h  $(INSTALL_DIR)/include


.PHONY:  build clean default install  $(libsubdir_list) $(binsubdir_list)


$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) $< $(COMMON_OBJS) -o $@ $(LIBS) $(SO_LIBS)

SO_LIBS=-Wl,-rpath,./build/core -L ../build/core -ltengine
LIBS+=-lpthread -ldl


