#ifndef __REF_CEIL_KERNEL_H__
#define __REF_CEIL_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ceil_param;

struct ceil_param
{
    int in_dim[4];
    int dim_size;
};

typedef int (*ref_ceil_t)(void* input, void* output, ceil_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_ceil_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
