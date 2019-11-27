BUILD_DIR?=$(shell pwd)/build
INSTALL_DIR?=$(shell pwd)/install
MAKEBUILD?=$(shell pwd)/makefile.build

export CC CXX CFLAGS  LD LDFLAGS CXXFLAGS COMMON_CFLAGS

default: $(LIB_HCL_SO)

INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include 
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../executor/include

CXXFLAGS+=


COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g  -fPIC  $(INC_DIR) -Werror

# To enable auth function
# CONFIG_AUTH_DEVICE=y

ifeq ($(CONFIG_AUTH_DEVICE),y)
    COMMON_CFLAGS+= -DCONFIG_AUTH_DEVICE -I$(shell pwd)/auth

ifeq ($(CONFIG_INTERN_RELEASE),y)
    COMMON_CFLAGS+=-DCONFIG_INTERN_RELEASE
endif

ifeq ($(CONFIG_INTERN_TRIAL),y)
    COMMON_CFLAGS+=-DCONFIG_INTERN_TRIAL
endif

ifdef CONFIG_INTERN_RELEASE
ifdef CONFIG_INTERN_TRIAL
    $(error "ERROR: both CONFIG_INTERN_RELEASE and CONFIG_INTERN_TRIAL defined, please check auth/auth/config")
endif

ifdef CONFIG_AUTHENICATION 
    $(error "ERROR: both CONFIG_INTERN_RELEASE and CONFIG_AUTHENICATIOIN defined, please check auth/auth/config")
endif
endif

ifneq ($(CONFIG_OPT_CFLAGS),)
    COMMON_CFLAGS+=-O3 -funroll-loops
endif

endif

ARM_BLOB=$(BUILD_DIR)/arm-builtin.o
LIB_HCL_SO?=$(BUILD_DIR)/../libhclcpu.so


LIB_SUB_DIRS+=../executor/operator lib/

ifeq ($(CONFIG_ARCH_ARM64),y)
	COMMON_CFLAGS+= -DCONFIG_ARCH_ARM64=1
endif

ifeq ($(CONFIG_ARCH_BLAS),y)
	COMMON_CFLAGS+=-DCONFIG_ARCH_BLAS=1
	COMMON_CFLAGS+=-I$(OPENBLAS_CFLAGS)
	LIB_LDFLAGS+=-L$(OPENBLAS_LIB) -lopenblas
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
	COMMON_CFLAGS+= -DCONFIG_ARCH_ARM32=1
	CC+= -march=armv7-a -mfpu=neon
	CXX+=-march=armv7-a -mfpu=neon
endif

ARM_OBJS =$(addprefix $(BUILD_DIR)/, $(foreach f,$(LIB_SUB_DIRS),$(f)/built-in.o))

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






