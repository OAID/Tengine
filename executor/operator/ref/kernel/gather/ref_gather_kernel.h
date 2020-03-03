#ifndef __GATHER_KERNEL_H__
#define __GATHER_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gather_param
{
    int in_shape[4];    // the dim of the input
    int axis;
    int indices_num;
    int dim_size;    
};

typedef int (*ref_gather_t)(void* input,void* input_indices,void* output, const struct gather_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_gather_fp32.c"
#endif

// #ifdef CONFIG_KERNEL_FP16
// #include "ref_slice_fp16.c"
// #endif

// #ifdef CONFIG_KERNEL_INT8
// #include "ref_slice_int8.c"
// #endif

// #ifdef CONFIG_KERNEL_UINT8
// #include "ref_slice_uint8.c"
// #endif

#ifdef __cplusplus
}
#endif

#endif
