include makefile.config

MAKEBUILD=$(shell pwd)/scripts/makefile.build

BUILD_DIR=$(shell pwd)/build
INSTALL_DIR=$(shell pwd)/install
TOP_DIR=$(shell pwd)

export INSTALL_DIR MAKEBUILD TOP_DIR

LIB_SUB_DIRS=core serializer operator  executor
APP_SUB_DIRS=tests

ifeq ($(CONFIG_ARCH_ARM64),y)
    export CONFIG_ARCH_ARM64

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

$(SUB_DIRS):
	@$(MAKE) -C $@  BUILD_DIR=$(BUILD_DIR)/$@ $(MAKECMDGOALS)


distclean:
	find . -name $(BUILD_DIR) | xargs rm -rf
	find . -name $(INSTALL_DIR) | xargs rm -rf

.PHONY: clean install $(SUB_DIRS) build
