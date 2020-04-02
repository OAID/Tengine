#ifndef __REVERSE_KERNEL_H__
#define __REVERSE_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct reverse_param
{
    int in_shape[4];    // the dim of the input
    int dim_size;
};

typedef int (*ref_reverse_t)(void* input, void* input_axis, void* output, const struct reverse_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_reverse_fp32.c"
#endif

// #ifdef CONFIG_KERNEL_FP16
// #include "refreverse_fp16.c"
// #endif

// #ifdef CONFIG_KERNEL_INT8
// #include "refreverse_int8.c"
// #endif

// #ifdef CONFIG_KERNEL_UINT8
// #include "refreverse_uint8.c"
// #endif

#ifdef __cplusplus
}
#endif

#endif
