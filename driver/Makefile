driver-$(CONFIG_ACL_GPU)+=acl_graph/

obj-y+=cpu/ plugin/  $(driver-y)


INC_DIR+=-I$(shell pwd)/include 
INC_DIR+=-I$(shell pwd)/../include 
INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/../executor/include 

CXXFLAGS+=

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -fPIC  $(INC_DIR) 

ifeq ($(CONFIG_ARCH_ARM64),y)
    COMMON_CFLAGS+= -DCONFIG_ARCH_ARM64=1
endif

ifeq ($(CONFIG_ACL_GPU),y)
    COMMON_CFLAGS+= -DCONFIG_ACL_GPU=1
endif

ifeq ($(CONFIG_ARCH_ARM32),y)
    COMMON_CFLAGS+= -DCONFIG_ARCH_ARM32=1
endif

install:



