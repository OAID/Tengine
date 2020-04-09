#ifndef __SPARSETODENSE_KERNEL_H__
#define __SPARSETODENSE_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "graph.hpp"

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sparsetodense_param 
{
    float default_value;
    int indices_dim_size;
    int output_dim_size;
    int sparse_values_size;
    int indices_shape[1];
};

typedef int (*ref_sparsetodense_t)(void* input, void* outout_shape, void* sparse_values, void* output, struct sparsetodense_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_sparsetodense_fp32.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
