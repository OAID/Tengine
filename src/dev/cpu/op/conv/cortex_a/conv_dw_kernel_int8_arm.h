/*
 * Author: 1091545398@qq.com
 */

#ifndef TENGINLITE_CONV_DW_KERNEL_INT8_ARM_H
#define TENGINLITE_CONV_DW_KERNEL_INT8_ARM_H

#include "tengine_ir.h"
#include "convolution_param.h"

int conv_dw_int8_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                 struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity) __attribute__((weak));

#endif    // TENGINLITE_CONV_DW_KERNEL_INT8_ARM_H
