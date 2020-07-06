#ifndef _PRELU_KERNEL_ARM_H_
#define _PRELU_KERNEL_ARM_H_

#include "tengine_ir.h"

int prelu_kernel_run(float* input, float* output, int dim0, int dim1, int dim2, int dim3, float* slope, int layout,
                     int num_thread) __attribute__((weak));

#endif
