obj-$(CONFIG_ARCH_ARM64)+=arm64/
obj-y+=ref/
obj-y+=common/
obj-y+=init.o

COMMON_CFLAGS+=-I$(shell pwd)/include
ifeq ($(CONFIG_ARCH_BLAS),y)
	COMMON_CFLAGS+=-I/usr/include/openblas
endif

#below are examples to build with pre-compiled object
#prebuilt-obj-$(CONFIG_ARCH_ARM64)+=arm.o
#prebuilt-obj-y+=ref/built-in.o
