#ifndef __REF_NOOP_KERNEL_H__
#define __REF_NOOP_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ref_noop_t)(float scale, int zero_point);

#ifdef CONFIG_KERNEL_FP32
#include "ref_noop_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_noop_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_noop_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_noop_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
