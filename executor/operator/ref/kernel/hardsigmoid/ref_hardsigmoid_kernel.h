#ifndef __REF_HARDSIGMOID_KERNEL_H__
#define __REF_HARDSIGMOID_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_hardsigmoid_t)(void* in_data, void* out_data, int size, int alpha, int beta,float scale, int zero_point);

#ifdef CONFIG_KERNEL_FP32
#include "ref_hardsigmoid_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_hardsigmoid_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_hardsigmoid_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_hardsigmoid_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
