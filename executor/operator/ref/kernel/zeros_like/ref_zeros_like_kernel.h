#ifndef __REF_ZEROS_LIKE_KERNEL_H__
#define __REF_ZEROS_LIKE_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct zeros_like_param;

struct zeros_like_param
{
    int in_dim[4];
    int dim_size;
};

typedef int (*ref_zeros_like_t)(void* input, void* output, zeros_like_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_zeros_like_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
