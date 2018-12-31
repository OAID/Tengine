obj-y+=operator/
obj-y+=lib/
obj-y+=engine/
obj-y+=plugin/

INC_DIR+=-I$(shell pwd)/../core/include
INC_DIR+=-I$(shell pwd)/../include -I$(shell pwd)/../operator/include 
INC_DIR+=-I$(shell pwd)/include 

COMMON_CFLAGS+=$(CONFIG_OPT_CFLAGS)
COMMON_CFLAGS+= -Wall -g -fPIC  $(INC_DIR)  -Werror

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


install:


