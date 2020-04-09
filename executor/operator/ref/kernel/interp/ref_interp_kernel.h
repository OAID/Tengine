#ifndef __REF_INTERP_KERNEL_H__
#define __REF_INTERP_KERNEL_H__

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
#define INTERP_MIN(a, b) ((a) < (b) ? (a) : (b))
struct interp_param;

struct interp_param
{
    float width_scale;
    float height_scale;
    int batch_number;
    int inc;
    int inh;
    int inw;
    int output_width;
    int output_height;
};

typedef int (*ref_interp_t)(void* input,void* output, interp_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_interp_fp32.c"
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