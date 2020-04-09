#ifndef __REF_EMBED_KERNEL_H__
#define __REF_EMBED_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_embed_t)(void* in_data, void* out_data, void* weight_data, void* bias_data, int input_dim, int num_output, int size, int bias_term, float scale, int zero_point);

#ifdef CONFIG_KERNEL_FP32
#include "ref_embed_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_embed_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_embed_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_embed_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
