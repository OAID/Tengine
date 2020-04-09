#ifndef __REF_BIAS_KERNEL_H__
#define __REF_BIAS_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_bias_t)(void* in_data, void* out_data, void* bias_data,int size,int c, float scale, int zero_point);

#ifdef CONFIG_KERNEL_FP32
#include "ref_bias_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_bias_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_bias_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_bias_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
