###     cross compile for ARM64
#CROSS_COMPILE=aarch64-linux-gnu-
###     cross compile for ARM32
#CROSS_COMPILE=arm-linux-gnueabihf-
SYSROOT:=$(shell pwd)/sysroot/ubuntu_rootfs

ifneq ($(CROSS_COMPILE),"aarch64-linux-gnu-")
   SYSROOT_FLAGS:=--sysroot=$(SYSROOT) 
   SYSROOT_LDFLAGS:=-L/usr/lib/aarch64-linux-gnu -L/lib/aarch64-linux-gnu
   PKG_CONFIG_PATH:=$(SYSROOT)/usr/lib/aarch64-linux-gnu/pkgconfig
   export PKG_CONFIG_PATH
endif
ifneq ($(CROSS_COMPILE),"arm-linux-gnueabihf-")
   SYSROOT_FLAGS:=--sysroot=$(SYSROOT)32 
   SYSROOT_LDFLAGS:=-L/usr/lib/arm-linux-gnueabihf -L/lib/arm-linux-gnueabihf
   PKG_CONFIG_PATH:=$(SYSROOT)/usr/lib/arm-linux-gnueabihf/pkgconfig
   export PKG_CONFIG_PATH
endif

CC=$(CROSS_COMPILE)gcc -std=gnu99 $(SYSROOT_FLAGS)
CXX=$(CROSS_COMPILE)g++ -std=c++11 $(SYSROOT_FLAGS)
LD=$(CROSS_COMPILE)g++ $(SYSROOT_FLAGS) $(SYSROOT_LDFLAGS)
AR=$(CROSS_COMPILE)ar

BUILT_IN_LD=$(CROSS_COMPILE)ld

GIT_COMMIT_ID=$(shell git rev-parse HEAD)

COMMON_CFLAGS+=-Wno-ignored-attributes -Werror

export CC CXX CFLAGS BUILT_IN_LD LD LDFLAGS CXXFLAGS COMMON_CFLAGS 
export GIT_COMMIT_ID

include makefile.config

MAKEBUILD=$(shell pwd)/scripts/makefile.build

BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
TOP_DIR=$(shell pwd)

export INSTALL_DIR MAKEBUILD TOP_DIR


LIB_SUB_DIRS=core operator executor serializer driver


LIB_SO=$(BUILD_DIR)/libtengine.so
LIB_A=$(BUILD_DIR)/libtengine.a

LIB_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

APP_SUB_DIRS+=tools

ifeq ($(CONFIG_FRAMEWORK_WRAPPER),y)
    APP_SUB_DIRS+=wrapper
endif

#APP_SUB_DIRS+=internal
APP_SUB_DIRS+=tests

ifeq ($(CONFIG_ARCH_ARM64),y)
    export CONFIG_ARCH_ARM64
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
    export CONFIG_ARCH_ARM32
	COMMON_CFLAGS+=-mfp16-format=ieee -mfpu=neon-fp16
endif

ifeq ($(CONFIG_ARCH_BLAS),y)
    export CONFIG_ARCH_BLAS
endif


ifeq ($(CONFIG_CAFFE_REF),y)
    export CONFIG_CAFFE_REF
    export CAFFE_ROOT
endif

ifneq ($(CONFIG_OPT_CFLAGS),)
    export CONFIG_OPT_CFLAGS
endif

ifeq ($(CONFIG_ACL_GPU),y)
    export CONFIG_ACL_GPU
    export ACL_ROOT
endif

ifeq ($(CONFIG_CAFFE_SERIALIZER),y)
    export CONFIG_CAFFE_SERIALIZER
endif
ifeq ($(CONFIG_ONNX_SERIALIZER),y)
    export CONFIG_ONNX_SERIALIZER
endif
ifeq ($(CONFIG_MXNET_SERIALIZER),y)
    export CONFIG_MXNET_SERIALIZER
endif

ifeq ($(CONFIG_TF_SERIALIZER),y)
    export CONFIG_TF_SERIALIZER
endif

ifeq ($(CONFIG_TENGINE_SERIALIZER),y)
    export CONFIG_TENGINE_SERIALIZER
endif

SUB_DIRS=$(LIB_SUB_DIRS) $(APP_SUB_DIRS)

default: $(LIB_SO) $(APP_SUB_DIRS) 

build : default


clean: $(SUB_DIRS)

install: $(APP_SUB_DIRS)
	@mkdir -p $(INSTALL_DIR)/include $(INSTALL_DIR)/lib
	cp -f core/include/tengine_c_api.h $(INSTALL_DIR)/include
	cp -f core/include/cpu_device.h $(INSTALL_DIR)/include
	cp -f core/include/tengine_test_api.h $(INSTALL_DIR)/include
	cp -f $(BUILD_DIR)/libtengine.so $(INSTALL_DIR)/lib


ifeq ($(CONFIG_ACL_GPU),y)
    ACL_LIBS+=-Wl,-rpath,$(ACL_ROOT)/build/ -L$(ACL_ROOT)/build
    ACL_LIBS+= -larm_compute_core -larm_compute
    LIB_LDFLAGS+=$(ACL_LIBS) 
endif
ifeq ($(CONFIG_CAFFE_REF),y)
    CAFFE_LIBS+=-Wl,-rpath,$(CAFFE_ROOT)/build/lib -L$(CAFFE_ROOT)/build/lib -lcaffe
    CAFFE_LIBS+= -lprotobuf -lboost_system -lglog
    LIB_LDFLAGS+=$(CAFFE_LIBS) 
endif


$(LIB_OBJS): $(LIB_SUB_DIRS);

$(LIB_SO): $(LIB_OBJS)
	$(CC) -o $@ -shared -Wl,-Bsymbolic -Wl,-Bsymbolic-functions $(LIB_OBJS) $(LIB_LDFLAGS)

static: static_lib static_example

static_lib:
	@touch core/lib/compiler.cpp
	@export STATIC_BUILD=y && $(MAKE)
	$(AR) -crs $(LIB_A) $(LIB_OBJS)
	@rm $(BUILD_DIR)/libtengine.so

static_example: static_lib
	$(LD) -o $(BUILD_DIR)/test_tm  $(BUILD_DIR)/tests/bin/test_tm.o $(LIBS) -ltengine \
	      -ldl -lpthread  -static -L$(BUILD_DIR)
	@echo ; echo static example: $(BUILD_DIR)/test_tm  created

LIB_LDFLAGS+=-lpthread -lprotobuf -ldl

ifeq ($(CONFIG_ARCH_BLAS),y)
    LIB_LDFLAGS+=-lopenblas
endif

ifneq ($(MAKECMDGOALS),clean)
     $(APP_SUB_DIRS): $(LIB_SO)
endif   

$(LIB_SUB_DIRS):
	@$(MAKE) -C $@  -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

$(APP_SUB_DIRS):
	@$(MAKE) -C $@  BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


Makefile: makefile.config
	@touch Makefile
	@$(MAKE) clean

distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build
