#ifndef __REF_MVN_KERNEL_H__
#define __REF_MVN_KERNEL_H__

#include <stdint.h>
#include <math.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ref_mvn_param
{
    int input_n;
    int input_h;
    int input_w;
    int input_c;
    int across_channels;
    int normalize_variance;
    float eps;
    int layout;
    float in_scale;
    int in_zero;
    float out_scale;
    int out_zero;
    float scale_scale;
    int scale_zero;
};

typedef int (*ref_mvn_kernel_t)(void* input, void* output, const ref_mvn_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_mvn_fp32.c"
#endif
/*
#ifdef CONFIG_KERNEL_FP16
#include "ref_mvn_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_mvn_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_mvn_uint8.c"
#endif
*/
#ifdef __cplusplus
}
#endif

#endif
