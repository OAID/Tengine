/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haoluo@openailab.com
 */
#ifndef __CONV_DW_DILATION_KERNEL_ARM_H_
#define __CONV_DW_DILATION_KERNEL_ARM_H_

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "convolution_param.h"
#include "conv_dw_k5_k7_kernel_arm.h"

int conv_dw_dilation_run(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h, int input_w,
                         int channel, int pad, int activation, int num_thread)
{
    int channel_size = input_h * input_w;
    int mid_w = input_w - pad * 2;
    int mid_block_end = (mid_w & -4) + pad;
    int mid_end = mid_w + pad;
    int w = 0;
#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < channel; c++)
    {
        float* input_buf_c = input_buf + c * channel_size;
        float* output_buf_c = output_buf + c * channel_size;
        float* weight_buf_c = weight_buf + c * 9;
        float bias_c = bias ? bias[c] : 0;
        for (int h = 0; h < pad; h++)
        {
            for (w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
            }
            for (; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[6]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[7]), vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[8]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for (; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
            for (; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
        }
        for (int h = pad; h < input_h - pad; h++)
        {
            for (w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
            for (; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[0]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[1]), vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[2]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[6]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[7]), vld1q_f32(input_buf_c + (h + pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[8]),
                                  vld1q_f32(input_buf_c + (h + pad) * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for (; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                tmp += weight_buf_c[8] * input_buf_c[(h + pad) * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
            for (; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[6] * input_buf_c[(h + pad) * input_w + w - pad];
                tmp += weight_buf_c[7] * input_buf_c[(h + pad) * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
        }
        for (int h = input_h - pad; h < input_h; h++)
        {
            for (w = 0; w < pad; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
            for (; w < mid_block_end; w += 4)
            {
                float32x4_t tmp_4 = vdupq_n_f32(bias_c);

                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[0]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[1]), vld1q_f32(input_buf_c + (h - pad) * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[2]),
                                  vld1q_f32(input_buf_c + (h - pad) * input_w + w + pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[3]), vld1q_f32(input_buf_c + h * input_w + w - pad));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[4]), vld1q_f32(input_buf_c + h * input_w + w));
                tmp_4 = vmlaq_f32(tmp_4, vdupq_n_f32(weight_buf_c[5]), vld1q_f32(input_buf_c + h * input_w + w + pad));
                tmp_4 = vector_activation(tmp_4, activation);
                vst1q_f32(output_buf_c + h * input_w + w, tmp_4);
            }
            for (; w < mid_end; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[2] * input_buf_c[(h - pad) * input_w + w + pad];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                tmp += weight_buf_c[5] * input_buf_c[h * input_w + w + pad];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
            for (; w < input_w; w++)
            {
                float tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[(h - pad) * input_w + w - pad];
                tmp += weight_buf_c[1] * input_buf_c[(h - pad) * input_w + w];
                tmp += weight_buf_c[3] * input_buf_c[h * input_w + w - pad];
                tmp += weight_buf_c[4] * input_buf_c[h * input_w + w];
                output_buf_c[h * input_w + w] = elem_activation(tmp, activation);
                ;
            }
        }
    }

    return 0;
}

#endif
