obj-y+=fc/
obj-y+=conv/
obj-y+=act/
obj-y+=fused/
obj-y+=dropout.o
obj-y+=softmax.o
obj-y+=concat.o
obj-y+=lrn.o
obj-y+=scale.o
obj-y+=batch_norm.o
obj-y+=pooling.o
obj-y+=scale_neon.o
obj-y+=concat_neon.o
obj-y+=eltwise.o
obj-y+=slice.o
obj-y+=normalize.o

obj-y+=prelu.o
CXXFLAGS+=-I./include 
CXXFLAGS+=-O3 -funroll-loops

pooling_CXXFLAGS+=-Wno-strict-aliasing
