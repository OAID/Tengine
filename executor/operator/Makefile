obj-$(CONFIG_ARCH_ARM64)+=arm64/
obj-y+=ref/
obj-y+=common/

COMMON_CFLAGS+=-I$(shell pwd)/include

ifneq ($(CONFIG_OPT_CFLAGS),)
   COMMON_CFLAGS+=-O3 -funroll-loops
endif

#below are examples to build with pre-compiled object
#prebuilt-obj-$(CONFIG_ARCH_ARM64)+=arm.o
#prebuilt-obj-y+=ref/built-in.o
