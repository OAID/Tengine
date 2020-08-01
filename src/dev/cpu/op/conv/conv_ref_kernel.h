#ifndef _CONV_KERNEL_REF_H_
#define _CONV_KERNEL_REF_H_

#include "tengine_ir.h"
#include "convolution_param.h"

struct conv_priv_info
{
    void* interleave_buffer;
    void* im2col_buffer;
    void* p_input_max;
    void* p_kernel_max;
    int im2col_buffer_size;
    int interleave_buffer_size;
    int external_im2col_mem;
    int external_interleave_mem;
};

int conv_kernel_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* output_tensor,
                       struct conv_priv_info* info, struct conv_param* param) __attribute__((weak));

int conv_kernel_postrun(struct conv_priv_info* info) __attribute__((weak));

int conv_kernel_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                    struct ir_tensor* output_tensor, struct conv_priv_info* conv_info, struct conv_param* param,
                    int num_thread, int cpu_affinity) __attribute__((weak));

int conv_kernel_get_shared_mem_size(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                                    struct conv_param* param) __attribute__((weak));

int conv_kernel_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size) __attribute__((weak));

#endif
