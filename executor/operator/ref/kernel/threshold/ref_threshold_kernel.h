#ifndef __REF_THRESHOLD_KERNEL_H__
#define __REF_THRESHOLD_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_threshold_t)(void* out_data, void* in_data, float threshold, int size, float scale, int zero_point);

#ifdef CONFIG_KERNEL_FP32
#include "ref_threshold_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_threshold_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_threshold_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_threshold_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
