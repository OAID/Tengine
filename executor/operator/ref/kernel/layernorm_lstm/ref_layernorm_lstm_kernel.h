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
 * Copyright (c) 2019, Open AI Lab
 * Author: zpluo@openailab.com
 */

#ifndef __REF_LNLSTM_KERNEL_H__
#define __REF_LNLSTM_KERNEL_H__

#include <stdint.h>
#include <math.h>
#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Lnlstm_MAX(a, b) ((a) > (b) ? (a) : (b))
#define Lnlstm_MIN(a, b) ((a) < (b) ? (a) : (b))

const float LayerNormEpsilon = 1e-8;

// enum class FusedActivation{
//   kNone = 0,
//   kRelu,
//   kRelu1,
//   kRelu6,
//   kTanh,
//   kSignBit,
//   kSigmoid,
// };

struct lnlstm_param{
    float scale[2];
    int zero_point[2];
    int batch_size;
    int input_size;
    int cell_size;
    int output_size; 
    int output_true_size;
    int sequence_size; 
    const float* i2i_weights_data; 
    const float* i2c_weights_data;
    const float* i2f_weights_data;
    const float* i2o_weights_data;
    const float* igate_bias_data;
    const float* cgate_bias_data;
    const float* fgate_bias_data;
    const float* ogate_bias_data;
    const float* r2i_weights_data;
    const float* r2c_weights_data;
    const float* r2f_weights_data;
    const float* r2o_weights_data;
    const float* c2i_weights_data;
    const float* c2f_weights_data;
    const float* c2o_weights_data;
    const float* projection_weights_data;
    const float* projection_bias_data;
    float* icellstate_data;
    float* iactivationstate_data;
    const float* ilayer_norm_coefficients_data;
    const float* flayer_norm_coefficients_data;
    const float* clayer_norm_coefficients_data;
    const float* olayer_norm_coefficients_data;
    TEngine::FusedActivation fused_activation;
    float cell_clip;
    float proj_clip;
    float* input_gate_scratch;
    float* forget_gate_scratch;
    float* cell_scratch;
    float* output_gate_scratch;
};

void mytanh(float* data, float* output, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = std::tanh(data[i]);
    }
}

void sigmoid(float* data, float* output, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = 1 / (1 + exp(-data[i]));
    }
}

void matbatchvectorwiseproduct(const float* matrix, const float* vector, float* output, int batch, int rows, int cols)
{
    float* result = output;
    for(int i = 0; i < batch; i++)
    {
        const float* matrix_ptr = matrix;
        for(int j = 0; j < rows; j++)
        {
            double temp = 0.0;
            const float* vector_ptr = vector + i * cols;
            for(int p = 0; p < cols; p++)
            {   
                temp +=  *matrix_ptr++ * *vector_ptr++;
            }
            //std::cout <<temp <<std::endl;
            *result++ += temp;
        }
    }
}

void vectorbatchvectorwiseproductacc(const float* vector1, const float* vector2, float* output, int batch, int rows)
{
    float* result = output;
    const float* vector_cursor = vector2;
    for(int i = 0; i < batch; i++)
    {
        for(int j = 0; j < rows; j++)
        { 
            *result++ += vector1[j] * *vector_cursor++;
        }
    }
}

void vectorbatchvectorwiseproduct(const float* vector1, const float* vector2, float* output, int batch, int rows)
{
    float* result = output;
    const float* vector_cursor = vector2;
    for(int i = 0; i < batch; i++)
    {
        for(int j = 0; j < rows; j++)
        {
            double temp = 0.0;
            temp =  vector1[j] * *vector_cursor++;
            *result++ = temp;
        }
    }
}

void vectorvectorwiseproductacc(const float* vector1, const float* vector2, float*output, int size)
{
    float* result = output;
    for(int i = 0; i < size; i++)
    {
        double temp = 0.0;
        temp = vector1[i] * vector2[i];
        *result++ += temp;
    }
}

void vectorvectorwiseproduct(const float* vector1, const float* vector2, float*output, int size)
{
    float* result = output;
    for(int i = 0; i < size; i++)
    {
        double temp = 0.0;
        temp = vector1[i] * vector2[i];
        *result++ = temp;
    }
}

void vectoradd(const float* vector1, const float* vector2, float* output, int batch, int rows)
{
    float* result = output;
    for(int i = 0; i < batch; i++)
    {
        for(int j = 0; j < rows; j++)
        {
            *result++ = *vector1++ + vector2[j];
        }
    }

}

void vector1sub(const float* input, float* output, int size)
{
    for(int i = 0; i < size; i++)
    {
        output[i] = 1.0f - input[i];
    }
}

void vectorclip(const float* input, float* output, int size, float abs)
{
    for(int i = 0; i < 0; i++)
    {
        output[i] = std::max(std::min(input[i], abs), -abs);
    }
}

void vectorbatchvectorassign(const float* input, float* destvec, int batch, int size)
{
    for(int i = 0; i < batch; i++)
    {
        memcpy(destvec + i * size, input, size * sizeof(float));
    }
}

void layer_norm_nhwc(const float* input, float* output, int cell_size, int batch_size)
{
    float* result = output;
    for(int i = 0; i < batch_size; i++)
    {
        float sum    = 0.0f;
        float sum_sq = 0.0f;
        for(int j = 0; j < cell_size; j++)
        {
            sum += input[j];
            sum_sq += input[j] * input[j];
        }
        const float mean = sum / cell_size;
        const float variance = sum_sq / cell_size - mean * mean;
        float std_idv = 0;
        if(0 == variance)
        {
            std_idv = 1.0f / sqrt(LayerNormEpsilon);
        }
        else
        {
            std_idv = 1.0f / sqrt(variance);
        }
        for(int j = 0; j < cell_size; j++)
        {
            *result++ = (input[j] - mean) * std_idv;
        }
        input  += cell_size;
    }
}   

void dump_scratch(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%.7f ",data[i]);
    }
    std::cout << std::endl;
}

void activationswitch(float* input, float* output,int batch_size, int cell_size, TEngine::FusedActivation activationtype)
{
    switch(activationtype)
    {
        case TEngine::FusedActivation::kSigmoid:
        {
            sigmoid(input, output, batch_size * cell_size);
            break;
        }
        case TEngine::FusedActivation::kTanh:
        {
            mytanh(input, output, batch_size * cell_size);
            break;
        }
        default:{
            break;
        }
    }
}
void DoLayerNormLSTM(float* input, float* output, 
                         const float* i2i_weights_data, const float* i2c_weights_data, const float* i2f_weights_data, const float* i2o_weights_data, 
                         const float* r2i_weights_data, const float* r2c_weights_data, const float* r2f_weights_data, const float* r2o_weights_data, 
                         const float* c2i_weights_data, const float* c2f_weights_data, const float* c2o_weights_data, 
                         const float* igate_bias_data,  const float* cgate_bias_data,  const float* fgate_bias_data,  const float* ogate_bias_data,
                         const float* projection_bias_data, const float* projection_weights_data,
                         float* icellstate_data, float* iactivationstate_data,
                         const float* ilayer_norm_coefficients_data,
                         const float* flayer_norm_coefficients_data, 
                         const float* clayer_norm_coefficients_data, 
                         const float* olayer_norm_coefficients_data,
                         int batch_size, int cell_size, int input_size, int output_size, int output_real_size,
                         TEngine::FusedActivation activationtype, float cell_clip, float proj_clip,float* input_gate_scratch,
                         float* forget_gate_scratch,float* cell_scratch,float* output_gate_scratch)
{
    const bool use_cifg = (i2i_weights_data == nullptr);
    const bool use_peephole = (c2o_weights_data != nullptr);
    if(!use_cifg)
    {
        memset(input_gate_scratch, 0, batch_size * cell_size * sizeof(float));
    }
    memset(forget_gate_scratch, 0, batch_size * cell_size * sizeof(float));
    memset(cell_scratch, 0, batch_size * cell_size * sizeof(float));
    memset(output_gate_scratch, 0, batch_size * cell_size * sizeof(float));
    //memset(icellstate_data, 0, batch_size * cell_size * sizeof(float));
    //memset(iactivationstate_data, 0, batch_size * output_size * sizeof(float));
    if(!use_cifg)
    {
        matbatchvectorwiseproduct(i2i_weights_data, input, input_gate_scratch, batch_size, cell_size, input_size);
    }
    //dump_scratch(const_cast<float*>(i2f_weights_data), cell_size*input_size);
    matbatchvectorwiseproduct(i2f_weights_data, input, forget_gate_scratch, batch_size, cell_size, input_size);
    matbatchvectorwiseproduct(i2c_weights_data, input, cell_scratch, batch_size, cell_size, input_size);
    matbatchvectorwiseproduct(i2o_weights_data, input, output_gate_scratch, batch_size, cell_size, input_size);

    //std::cout<<"first deal."<<std::endl;
    //dump_scratch(input_gate_scratch, cell_size*batch_size);
    //dump_scratch(forget_gate_scratch, cell_size*batch_size);
    //dump_scratch(cell_scratch, cell_size*batch_size);
    //dump_scratch(output_gate_scratch, cell_size*batch_size);
    //dump_scratch(icellstate_data, cell_size*batch_size);
    if(!use_cifg)
    {
        matbatchvectorwiseproduct(r2i_weights_data, iactivationstate_data, input_gate_scratch, batch_size, cell_size, output_size);
    }
    matbatchvectorwiseproduct(r2f_weights_data, iactivationstate_data, forget_gate_scratch, batch_size, cell_size, output_size);
    matbatchvectorwiseproduct(r2c_weights_data, iactivationstate_data, cell_scratch, batch_size, cell_size, output_size);
    matbatchvectorwiseproduct(r2o_weights_data, iactivationstate_data, output_gate_scratch, batch_size, cell_size, output_size);
    
    //update input gate
    if(!use_cifg)
    {
        if(use_peephole)
        {
            //dump_scratch(icellstate_data, cell_size*batch_size);
            vectorbatchvectorwiseproductacc(c2i_weights_data, icellstate_data, input_gate_scratch, batch_size, cell_size);
        }
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        layer_norm_nhwc(input_gate_scratch, input_gate_scratch, cell_size, batch_size);
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        vectorbatchvectorwiseproduct(ilayer_norm_coefficients_data, input_gate_scratch, input_gate_scratch, batch_size, cell_size);
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        vectoradd(input_gate_scratch, igate_bias_data, input_gate_scratch, batch_size, cell_size);
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
        sigmoid(input_gate_scratch, input_gate_scratch, cell_size * batch_size);
        //dump_scratch(input_gate_scratch, cell_size*batch_size);
    }
    //std::cout<<"update input gate done."<<std::endl;
    //dump_scratch(input_gate_scratch, cell_size*batch_size);
    //update forget gate
    if(use_peephole)
    {
        vectorbatchvectorwiseproductacc(c2f_weights_data, icellstate_data, forget_gate_scratch, batch_size, cell_size);
    }
    layer_norm_nhwc(forget_gate_scratch, forget_gate_scratch, cell_size, batch_size);
    vectorbatchvectorwiseproduct(flayer_norm_coefficients_data, forget_gate_scratch, forget_gate_scratch, batch_size, cell_size);
    vectoradd(forget_gate_scratch, fgate_bias_data, forget_gate_scratch, batch_size, cell_size);
    sigmoid(forget_gate_scratch, forget_gate_scratch, cell_size * batch_size);
    //std::cout<<"update forget gate."<<std::endl;
    //dump_scratch(forget_gate_scratch, cell_size*batch_size);
    //update cell
    vectorvectorwiseproduct(forget_gate_scratch, icellstate_data, icellstate_data,cell_size * batch_size);
    layer_norm_nhwc(cell_scratch, cell_scratch, cell_size, batch_size);
    vectorbatchvectorwiseproduct(clayer_norm_coefficients_data, cell_scratch, cell_scratch, batch_size, cell_size);
    vectoradd(cell_scratch, cgate_bias_data, cell_scratch, batch_size, cell_size);
    activationswitch(cell_scratch, cell_scratch, batch_size , cell_size, activationtype);
    if(use_cifg)
    {
        //std::cout << "use_cifg."<<std::endl;
        vector1sub(forget_gate_scratch, forget_gate_scratch, batch_size * cell_size);
        vectorvectorwiseproductacc(cell_scratch, forget_gate_scratch, icellstate_data, batch_size * cell_size);
    }
    else
    {
        vectorvectorwiseproductacc(cell_scratch, input_gate_scratch, icellstate_data, batch_size * cell_size);
    }
    //dump_scratch(icellstate_data, cell_size*batch_size);
    if(cell_clip > 0.0)
    {
        vectorclip(cell_scratch, cell_scratch, batch_size * cell_size, cell_clip);
    }
    //std::cout<<"update cell."<<std::endl;
    //dump_scratch(cell_scratch, cell_size*batch_size);
    //update output gate
    //std::cout<<"before update output gate."<<std::endl;
    //dump_scratch(output_gate_scratch, cell_size*batch_size);
    if(use_peephole)
    {
        vectorbatchvectorwiseproductacc(c2o_weights_data, icellstate_data, output_gate_scratch, batch_size, cell_size);
    }
    //std::cout<<"update output gate."<<std::endl;
    //dump_scratch(const_cast<float*>(c2o_weights_data), cell_size);
    layer_norm_nhwc(output_gate_scratch, output_gate_scratch, cell_size, batch_size);   
    vectorbatchvectorwiseproduct(olayer_norm_coefficients_data, output_gate_scratch, output_gate_scratch, batch_size, cell_size);
    vectoradd(output_gate_scratch, ogate_bias_data, output_gate_scratch, batch_size, cell_size);
    sigmoid(output_gate_scratch, output_gate_scratch, batch_size * cell_size);
    activationswitch(icellstate_data, cell_scratch, batch_size, cell_size, activationtype);
    vectorvectorwiseproduct(output_gate_scratch, cell_scratch, output_gate_scratch, batch_size * cell_size);
    const bool use_projection_weight = (projection_weights_data != nullptr);
    const bool use_projection_bias   = (projection_bias_data != nullptr);     

    if(output_real_size == output_size)
    {
        if(use_projection_weight)
        {
            if(use_projection_bias)
            {
                vectorbatchvectorassign(projection_bias_data, output, batch_size, output_size);
            }
            else
            {
                memset(output, 0, batch_size * output_size * sizeof(float));
            }
            matbatchvectorwiseproduct(projection_weights_data, output_gate_scratch, output, batch_size, output_size, cell_size);
            if(proj_clip > 0.0)
            {
                vectorclip(output, output, output_size * batch_size, proj_clip);
            }
            
        }
        else
        {
            memcpy(output, output_gate_scratch, batch_size * output_size * sizeof(float));
        }
        memcpy(iactivationstate_data, output, batch_size * output_size * sizeof(float));
        //dump_scratch(icellstate_data, batch_size * cell_size);
        //dump_scratch(iactivationstate_data, batch_size * output_size);
    }
    else
    {
        if(use_projection_weight)
        {
            if(use_projection_bias)
            {
                for(int n = 0; n < batch_size; n++)
                {
                    memcpy(output + n * output_real_size, projection_bias_data, output_real_size);
                }

            }
            else
            {
                for(int n = 0; n < batch_size; n++)
                {
                    memset(output + n * output_real_size, 0, output_real_size * sizeof(float));
                }
            }
            
            for(int n = 0; n < batch_size; n++)
            {
                matbatchvectorwiseproduct(projection_weights_data, output_gate_scratch + n * cell_size, 
                                            output + n * output_real_size, 1, output_real_size, cell_size);
                if(proj_clip > 0.0)
                {
                    vectorclip(output, output, output_size * batch_size, proj_clip);
                }
            }
        }
        else
        {
            for(int n = 0; n < batch_size; n++)
            {
                memcpy(output + n * output_real_size, output_gate_scratch + n * output_size, output_size);
            }
        }
        for(int n = 0; n < batch_size; n++)
        {
            memcpy(iactivationstate_data + n * output_size, output + n * output_real_size, output_size);
        }
        
    }
    
}

void getsinglesequence(float* input, float* output, int batch_size,int sequence_size, int per_sequence_size, int sequence_num)
{
    int per_batch_num = per_sequence_size * sequence_size;
    for(int i = 0; i < batch_size; i++)
    {
        for(int j = 0; j < per_sequence_size; j++)
        {
            output[i * per_sequence_size + j] = input[per_batch_num * i + sequence_num * per_sequence_size + j];
        }
    }
}

void mixsequence(float* input, float*output, int batch_size, int sequence_size, int per_sequence_size, int sequence_num)
{
    int per_batch_num = per_sequence_size * sequence_size;
    for(int i = 0; i < batch_size; i++)
    {
        for(int j = 0; j < per_sequence_size; j++)
        {
            output[per_batch_num * i + sequence_num * per_sequence_size + j] = input[i * per_sequence_size + j];    
        }
    }
}
    
typedef int (*ref_lnlstm_t)(void* in_data, void* out_data, lnlstm_param* param);

#ifdef CONFIG_KERNEL_FP32
#include "ref_layernorm_lstm_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_layernorm_lstm_fp16.c"
#endif

#ifdef CONFIG_KERNEL_INT8
#include "ref_layernorm_lstm_int8.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_layernorm_lstm_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
