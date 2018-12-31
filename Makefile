###     cross compile for ARM64
#CROSS_COMPILE=aarch64-linux-gnu-
###     cross compile for ARM32
#CROSS_COMPILE=arm-linux-gnueabihf-
SYSROOT:=$(shell pwd)/sysroot/ubuntu_rootfs

ifeq ($(CROSS_COMPILE),aarch64-linux-gnu-)
   SYSROOT_FLAGS:=--sysroot=$(SYSROOT) 
   SYSROOT_LDFLAGS:=-L/usr/lib/aarch64-linux-gnu -L/lib/aarch64-linux-gnu
   PKG_CONFIG_PATH:=$(SYSROOT)/usr/lib/aarch64-linux-gnu/pkgconfig
   export PKG_CONFIG_PATH
endif
ifeq ($(CROSS_COMPILE),arm-linux-gnueabihf-)
   SYSROOT_FLAGS:=--sysroot=$(SYSROOT)32 
   SYSROOT_LDFLAGS:=-L/usr/lib/arm-linux-gnueabihf -L/lib/arm-linux-gnueabihf
   PKG_CONFIG_PATH:=$(SYSROOT)32/usr/lib/arm-linux-gnueabihf/pkgconfig
   export PKG_CONFIG_PATH
endif

CC=$(CROSS_COMPILE)gcc -std=gnu99 $(SYSROOT_FLAGS)
CXX=$(CROSS_COMPILE)g++ -std=c++11 $(SYSROOT_FLAGS)
LD=$(CROSS_COMPILE)g++ $(SYSROOT_FLAGS) $(SYSROOT_LDFLAGS)
AR=$(CROSS_COMPILE)ar

BUILT_IN_LD=$(CROSS_COMPILE)ld

GIT_COMMIT_ID=$(shell git rev-parse HEAD)

COMMON_CFLAGS+=-Wno-ignored-attributes -Werror -g

export CC CXX CFLAGS BUILT_IN_LD LD LDFLAGS CXXFLAGS COMMON_CFLAGS 
export GIT_COMMIT_ID

MAKEFILE_CONFIG=$(shell pwd)/makefile.config
MAKEBUILD=$(shell pwd)/scripts/makefile.build

include $(MAKEFILE_CONFIG)

BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
TOP_DIR=$(shell pwd)

export INSTALL_DIR MAKEBUILD TOP_DIR MAKEFILE_CONFIG


LIB_SUB_DIRS=core operator executor serializer driver model_src


LIB_SO=$(BUILD_DIR)/libtengine.so
LIB_A=$(BUILD_DIR)/libtengine.a

LIB_OBJS=$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

APP_SUB_DIRS+=tools

ifeq ($(CONFIG_FRAMEWORK_WRAPPER),y)
    APP_SUB_DIRS+=wrapper
endif

APP_SUB_DIRS+=tests


ifeq ($(CONFIG_ARCH_ARM32),y)
	COMMON_CFLAGS+=-march=armv7-a -mfpu=neon -mfp16-format=ieee -mfpu=neon-fp16
endif

ifeq ($(CONFIG_FLOAT16),y)
	COMMON_CFLAGS+=-DCONFIG_FLOAT16
endif

ifeq ($(CONFIG_LEGACY_API),y)
	COMMON_CFLAGS+=-DCONFIG_LEGACY_API
endif


SUB_DIRS=$(LIB_SUB_DIRS) $(APP_SUB_DIRS)

default: $(LIB_SO) $(APP_SUB_DIRS) 

build : default


clean: $(SUB_DIRS)

install: $(APP_SUB_DIRS)
	@mkdir -p $(INSTALL_DIR)/include $(INSTALL_DIR)/lib
	cp -f core/include/tengine_c_api.h $(INSTALL_DIR)/include
	cp -f core/include/tengine_c_compat.h $(INSTALL_DIR)/include
	cp -f core/include/cpu_device.h $(INSTALL_DIR)/include
	cp -f core/include/tengine_test_api.h $(INSTALL_DIR)/include
	cp -f $(BUILD_DIR)/libtengine.so $(INSTALL_DIR)/lib


ifeq ($(CONFIG_ACL_GPU),y)
    ACL_LIBS+=-Wl,-rpath,$(ACL_ROOT)/build/ -L$(ACL_ROOT)/build
    ACL_LIBS+= -larm_compute_core -larm_compute
    LIB_LDFLAGS+=$(ACL_LIBS) 
endif


$(LIB_OBJS): $(LIB_SUB_DIRS);

#special handling for model_src

MODEL_C_SRC=$(wildcard model_src/*.c model_src/*.cpp model_src/*.S)

ifeq ($(MODEL_C_SRC),)
    REAL_LIB_OBJS=$(filter-out %/model_src/built-in.o,$(LIB_OBJS))
else
    REAL_LIB_OBJS=$(LIB_OBJS)
endif



$(LIB_SO): $(REAL_LIB_OBJS) 
	$(LD) -o $@ -shared -Wl,-Bsymbolic -Wl,-Bsymbolic-functions $(wildcard $(LIB_OBJS)) $(LIB_LDFLAGS)

static: static_lib static_example

static_lib:
	@touch core/lib/compiler.cpp
	@export STATIC_BUILD=y && $(MAKE)
	$(AR) -crs $(LIB_A) $(wildcard $(LIB_OBJS))
	@rm $(BUILD_DIR)/libtengine.so

static_example: static_lib
	$(LD) -o $(BUILD_DIR)/test_tm  $(BUILD_DIR)/tests/bin/test_tm.o $(LIBS) -ltengine \
	      -ldl -lpthread  -static -L$(BUILD_DIR) -lprotobuf -lblas -lpthread
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


Makefile: $(MAKEFILE_CONFIG)
	@touch Makefile
	@$(MAKE) clean

distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build
