
CC=$(CROSS_COMPILE)gcc -std=gnu99 
CXX=$(CROSS_COMPILE)g++ -std=c++11 
LD=$(CROSS_COMPILE)g++ 

AR=$(CROSS_COMPILE)ar

BUILT_IN_LD=$(CROSS_COMPILE)ld

GIT_COMMIT_ID=$(shell git rev-parse HEAD)

OPENBLAS_LIB=$(OPENBLAS_LIB_)

COMMON_CFLAGS+=-Wno-ignored-attributes -Werror -g

export CC CXX CFLAGS BUILT_IN_LD LD LDFLAGS CXXFLAGS COMMON_CFLAGS
export GIT_COMMIT_ID

MAKEBUILD=$(shell pwd)/scripts/makefile.build

BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
TOP_DIR=$(shell pwd)

export INSTALL_DIR MAKEBUILD TOP_DIR MAKEFILE_CONFIG

LIB_SUB_DIRS=core operator executor serializer driver model_src

LIB_SO=$(BUILD_DIR)/libtengine.so
LIB_A=$(BUILD_DIR)/libtengine.a
LIB_HCL_SO=$(BUILD_DIR)/libhclcpu.so
export LIB_HCL_SO

LIB_OBJS=$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

APP_SUB_DIRS+=tools
APP_SUB_DIRS+=benchmark
APP_SUB_DIRS+=tests

ifeq ($(CONFIG_ONLINE_REPORT),y)
	COMMON_CFLAGS+=-DENABLE_ONLINE_REPORT
	export CONFIG_ONLINE_REPORT		
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
	COMMON_CFLAGS+=-march=armv7-a -mfpu=neon -mfp16-format=ieee -mfpu=neon-fp16
        export CONFIG_ARCH_ARM32
endif

ifeq ($(CONFIG_ARCH_ARM64),y)
        export CONFIG_ARCH_ARM64
endif

COMMON_CFLAGS+=-DCONFIG_LEGACY_API

HCL_SUB_DIRS+=hclarm
LIB_HCL_OBJS=$(BUILD_DIR)/hclarm/arm-builtin.o

COMMON_CFLAGS+=-DCONFIG_KERNEL_FP32

ifeq ($(CONFIG_KERNEL_FP16),y)
    COMMON_CFLAGS+=-DCONFIG_KERNEL_FP16
endif

COMMON_CFLAGS+=-DCONFIG_KERNEL_INT8

COMMON_CFLAGS+=-DCONFIG_KERNEL_UINT8

SUB_DIRS=$(LIB_SUB_DIRS) $(APP_SUB_DIRS)

default: $(LIB_SO) $(LIB_A) $(LIB_HCL_SO) $(APP_SUB_DIRS) 

build : default


clean: $(SUB_DIRS) $(HCL_SUB_DIRS)

install: $(APP_SUB_DIRS) $(HCL_SUB_DIRS)
	@mkdir -p $(INSTALL_DIR)/include $(INSTALL_DIR)/lib
	cp -f core/include/tengine_c_api.h $(INSTALL_DIR)/include
	cp -f core/include/tengine_c_compat.h $(INSTALL_DIR)/include
	cp -f core/include/cpu_device.h $(INSTALL_DIR)/include
	cp -f $(BUILD_DIR)/libtengine.so $(INSTALL_DIR)/lib
	cp -f core/include/tengine_operations.h $(INSTALL_DIR)/include


$(LIB_OBJS): $(LIB_SUB_DIRS);

#special handling for model_src

MODEL_C_SRC=$(wildcard model_src/*.c model_src/*.cpp model_src/*.S)

ifeq ($(MODEL_C_SRC),)
    REAL_LIB_OBJS=$(filter-out %/model_src/built-in.o,$(LIB_OBJS))
else
    REAL_LIB_OBJS=$(LIB_OBJS)
endif

$(LIB_A): $(REAL_LIB_OBJS) 
	$(AR) crs $@  $(wildcard $(LIB_OBJS))


$(LIB_SO): $(REAL_LIB_OBJS) $(LIB_HCL_SO) 
	$(LD) -o $@ -shared -Wl,-Bsymbolic -Wl,-Bsymbolic-functions $(wildcard $(LIB_OBJS)) $(LIB_LDFLAGS) $ -L$(BUILD_DIR) -Wl,-rpath,\$$ORIGIN -Wl,-rpath-link=\$$ORIGIN

ifneq ( $(LIB_HCL_SO),)
     $(LIB_HCL_SO): $(HCL_SUB_DIRS);
else
     $(LIB_HCL_SO):
	
endif

static: static_lib static_example

static_lib:
	@touch core/lib/compiler.cpp
	@export STATIC_BUILD=y && $(MAKE)
	$(AR) -crs $(LIB_A) $(wildcard $(LIB_OBJS))
	@rm $(BUILD_DIR)/libtengine.so

static_example: static_lib
	$(LD) -o $(BUILD_DIR)/test_tm  $(BUILD_DIR)/tests/bin/test_tm.o $(LIBS) -ltengine \
	      -ldl -lpthread  -static -L$(BUILD_DIR)
	@echo ; echo static example: $(BUILD_DIR)/test_tm  created

LIB_LDFLAGS+=-lpthread -ldl

ifeq ($(CONFIG_ARCH_BLAS),y)
	export OPENBLAS_LIB OPENBLAS_CFLAGS
endif

ifneq ($(MAKECMDGOALS),clean)
     $(APP_SUB_DIRS): $(LIB_SO)
endif   

$(LIB_SUB_DIRS):
	@$(MAKE) -C $@  -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

$(APP_SUB_DIRS) $(HCL_SUB_DIRS):
	@$(MAKE) -C $@  BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


Makefile: $(MAKEFILE_CONFIG)
	@touch Makefile
	@$(MAKE) clean

distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build $(HCL_SUB_DIRS)
