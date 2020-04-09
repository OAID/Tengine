#ifndef __REF_INSTANCENORM_KERNEL_H__
#define __REF_INSTANCENORM_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_instancenorm_t)(void* input_data, void* output_data, void* gamma_data, void* beta_data, int size, int channels, int n, float eps, float scale, float zero_point, int layout);

#ifdef CONFIG_KERNEL_FP32
#include "ref_instancenorm_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_instancenorm_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_instancenorm_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_instancenorm_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
