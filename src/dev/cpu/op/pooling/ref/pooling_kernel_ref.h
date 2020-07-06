#ifndef __POOLING_KERNEL_REF_H__
#define __POOLING_KERNEL_REF_H__

#include "tengine_ir.h"
#include "pooling_param.h"

int pooling_kernel_ref_run(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                           struct pool_param* pool_param, int num_thread) __attribute__((weak));

#endif
