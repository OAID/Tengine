BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

default: $(LIB_HCL_SO)

include $(MAKEFILE_CONFIG)

INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../executor/include

CXXFLAGS+=


COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g  -fPIC  $(INC_DIR) -Werror

ifeq ($(CONFIG_INTERN_RELEASE),y)
    COMMON_CFLAGS+=-DCONFIG_INTERN_RELEASE
endif

ifeq ($(CONFIG_INTERN_TRIAL),y)
    COMMON_CFLAGS+=-DCONFIG_INTERN_TRIAL
endif

ifneq ($(CONFIG_OPT_CFLAGS),)
    COMMON_CFLAGS+=-O3 -funroll-loops
endif

ARM_BLOB=$(BUILD_DIR)/arm-builtin.o
LIB_HCL_SO?=$(BUILD_DIR)/../libhclcpu.so


LIB_SUB_DIRS+=../executor/operator  lib/

ifeq ($(CONFIG_ARCH_ARM64),y)
	COMMON_CFLAGS+= -DCONFIG_ARCH_ARM64=1
endif

ifeq ($(CONFIG_ARCH_BLAS),y)
	COMMON_CFLAGS+=-DCONFIG_ARCH_BLAS=1
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
	COMMON_CFLAGS+= -DCONFIG_ARCH_ARM32=1
	CC+= -march=armv7-a -mfpu=neon
	CXX+=-march=armv7-a -mfpu=neon
endif


ifeq ($(CONFIG_ACL_GPU),y)
	ACL_LIBS+=-Wl,-rpath,$(ACL_ROOT)/build/ -L$(ACL_ROOT)/build
	ACL_LIBS+= -larm_compute_core -larm_compute
	LIB_LDFLAGS+=$(ACL_LIBS)
endif

ARM_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

ifeq ($(CONFIG_ARCH_BLAS),y)
	LIB_LDFLAGS+=-lopenblas
endif

$(LIB_HCL_SO): $(ARM_BLOB)
	$(CC) -o $@ -shared -Wl,-Bsymbolic -Wl,-Bsymbolic-functions $(ARM_BLOB) $(LIB_LDFLAGS)

$(ARM_BLOB): $(ARM_OBJS)
	$(BUILT_IN_LD) -r -o $@ $(ARM_OBJS)


$(ARM_OBJS): $(LIB_SUB_DIRS);



build: default install


install:
	@mkdir -p $(INSTALL_DIR)/lib
	cp -f $(LIB_HCL_SO) $(INSTALL_DIR)/lib

$(LIB_SUB_DIRS):
	@$(MAKE) -C $@ -f $(MAKEBUILD) BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


clean: $(LIB_SUB_DIRS) 
	@rm -rf $(ARM_BLOB) $(LIB_HCL_SO)


.PHONY:  build clean default test install  $(LIB_SUB_DIRS) 






