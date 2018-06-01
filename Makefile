TOP_DIR=$(shell pwd)

export TOP_DIR

include makefile.config

ifdef SYSROOT
   SYSROOT_FLAGS:=--sysroot=${SYSROOT} -L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/blas
   SYSROOT_RPATH:=:/usr/lib/aarch64-linux-gnu/blas:/usr/lib/aarch64-linux-gnu/lapack
else
   SYSROOT_FLAGS:=
   SYSROOT_RPATH:=
endif

export CROSS_COMPILE SYSROOT SYSROOT_FLAGS SYSROOT_RPATH

MAKEBUILD=$(shell pwd)/scripts/makefile.build

BUILD_DIR=$(shell pwd)/build
INSTALL_DIR=$(shell pwd)/install

export INSTALL_DIR MAKEBUILD

LIB_SUB_DIRS=core serializer operator executor wrapper
LIB_SUB_DIRS+=driver
APP_SUB_DIRS=tests

ifeq ($(CONFIG_ARCH_ARM64),y)
    export CONFIG_ARCH_ARM64

ifeq ($(CONFIG_EVENT_EXECUTOR),y)
    export CONFIG_EVENT_EXECUTOR
    APP_SUB_DIRS+=devices
endif

else
   CONFIG_CAFFE_REF=y
endif

ifeq ($(CONFIG_CAFFE_REF),y)
    export CONFIG_CAFFE_REF
    export CAFFE_ROOT
endif


SUB_DIRS=$(LIB_SUB_DIRS) $(APP_SUB_DIRS)

default: $(APP_SUB_DIRS) 

build : default


clean: $(SUB_DIRS)

test: $(SUB_DIRS)


install: $(SUB_DIRS)
	@mkdir -p $(INSTALL_DIR)/etc
	@cp -f etc/config.example $(INSTALL_DIR)/etc

$(APP_SUB_DIRS): $(LIB_SUB_DIRS)

$(SUB_DIRS): ${SYSROOT}
	@$(MAKE) -C $@  BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)

distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build
