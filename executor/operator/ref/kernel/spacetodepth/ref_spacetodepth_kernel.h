#ifndef __REF_SPACETODEPTH_KERNEL_H__
#define __REF_SPACETODEPTH_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "graph.hpp"

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct spacetodepth_param 
{
    int size;

    int in_zero;
    int out_zero;
    float in_scale;
    float out_scale;
    int type;
};

typedef int (*ref_spacetodepth_t)(const void* in_data, const void* out_data, struct spacetodepth_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_spacetodepth_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
