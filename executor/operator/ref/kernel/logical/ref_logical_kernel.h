#ifndef __REF_LOGICAL_KERNEL_H__
#define __REF_LOGICAL_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct logical_param;

struct logical_param
{
    int type;
    int layout;
    int shape0[4];
    int shape1[4];
    int zero[3];
};

typedef int (*ref_logical_t)(void* input0, void* input1, void* output, logical_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_logical_fp32.c"
#endif

//#ifdef CONFIG_KERNEL_FP16
//#include "ref_logical_fp16.c"
//#endif

//#ifdef CONFIG_KERNEL_INT8
//#include "ref_logical_int8.c"
//#endif

//#ifdef CONFIG_KERNEL_UINT8
//#include "ref_logical_uint8.c"
//#endif

#ifdef __cplusplus
}
#endif

#endif