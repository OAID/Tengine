#ifndef __REF_ROUND_KERNEL_H__
#define __REF_ROUND_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct round_param;

struct round_param
{
    int in_dim[4];
    int dim_size;
};

typedef int (*ref_round_t)(void* input, void* output, round_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_round_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
