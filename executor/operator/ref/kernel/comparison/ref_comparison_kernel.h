#ifndef __REF_COMPARISON_KERNEL_H__
#define __REF_COMPARISON_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C"{
#endif

struct comparison_param;

struct comparison_param
{
    int type;
    int layout;
    int shape0[4];
    int shape1[4];
};

typedef int (*ref_comparison_t)(void* input0, void* input1, void* output, comparison_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_comparison_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif