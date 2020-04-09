#ifndef __REF_SPACETODEPTH_KERNEL_H__
#define __REF_SPACETODEPTH_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "graph.hpp"

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct depthtospace_param 
{
    int size;

    int in_zero;
    int out_zero;
    float in_scale;
    float out_scale;
    int type;
};

typedef int (*ref_depthtospace_t)(const void* in_data, const void* out_data, struct depthtospace_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_depthtospace_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
