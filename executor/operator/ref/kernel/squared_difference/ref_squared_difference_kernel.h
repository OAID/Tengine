#ifndef __REF_SQUARED_DIFFERENCE_KERNEL_H__
#define __REF_SQUARED_DIFFERENCE_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct squared_difference_param;

struct squared_difference_param
{
    int in_dim[4];
    int in_dim_size;
};

typedef int (*ref_squared_difference_t)(void* input1, void* input2, void* output, squared_difference_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_squared_difference_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif