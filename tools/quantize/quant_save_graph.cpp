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
* Copyright (c) 2020, OPEN AI LAB
* Author: hhchen@openailab.com
*/

#include <algorithm>
#include "quant_save_graph.hpp"
#include "compiler_fp16.h"

#include "operator/prototype/convolution_param.h"
#include "operator/prototype/pooling_param.h"
#include "operator/prototype/relu_param.h"

#ifdef _MSC_VER
#undef max
#undef min
#endif

void recursion_pass_through(struct graph* ir_graph, const char* layer_name, struct tensor* t,
                            std::tr1::unordered_map<std::string, int>& layer_used, std::tr1::unordered_map<std::string, float>& layer_scale,
                            std::tr1::unordered_map<std::string, float>& layer_zeropoint, std::tr1::unordered_map<std::string, bool>& layer_pass)
{
    if (layer_pass[t->name] == false && layer_used[t->name] < 2)
    {
        t->scale = layer_scale[layer_name];
        t->zero_point = layer_zeropoint[layer_name];
        layer_scale[t->name] = layer_scale[layer_name];
        layer_zeropoint[t->name] = layer_zeropoint[layer_name];

        uint32_t ir_node_idx = t->producer;
        struct node* t_node = ir_graph->node_list[ir_node_idx];

        std::string op_name = get_op_name_from_type(t_node->op.type);
        bool poolTrue = false;
        bool reluTrue = false;
        if (op_name == "Pooling")
        {
            struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
            if (pool_param->pool_method == 0)
                poolTrue = true;
        }
        else if (op_name == "ReLU")
        {
            struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
            if (relu_param->negative_slope == 0.f)
                reluTrue = true;
        }
        if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" || poolTrue || reluTrue)
        {
            struct tensor* t_in_tensor = ir_graph->tensor_list[t_node->input_tensors[0]];
            if (layer_scale[t->name] != 0)
            {
                if (t_in_tensor->tensor_type == TENSOR_TYPE_VAR || t_in_tensor->tensor_type == TENSOR_TYPE_INPUT)
                {
                    recursion_pass_through(ir_graph, t->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                }
            }
        }
        layer_pass[t->name] = true;
    }
}

int save_graph_u8_perlayer(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal)
{
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again\n");

    /* Step 1 : create graph, load tengine model xxx.tmfile */
    struct graph* ir_graph = (struct graph*)create_graph(nullptr, "tengine", model_file);
    if (nullptr == ir_graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again done.\n");

    std::tr1::unordered_map<std::string, float> layer_scale;
    std::tr1::unordered_map<std::string, float> layer_zeropoint;

    fprintf(stderr, "[Quant Tools Info]: Step 3, load calibration table file %s.\n", scale_file);
    /* Step 2 : set activation quant scale value into ir_tensor */
    if (nullptr != scale_file)
    {
        std::ifstream scales(scale_file);
        std::string line;
        while (std::getline(scales, line))
        {
            std::string layer_name;
            float scale_val = 0.f;
            float zero_point = 0.f;
            size_t last = 0;
            size_t index = line.find_first_of(' ', last);
            size_t idx = line.find_last_of(' ', line.size());
            layer_name = line.substr(last, index - last);
            last = index + 1;
            scale_val = atof((line.substr(last, line.size() - last)).c_str());
            zero_point = atof((line.substr(idx + 1, line.size())).c_str());

            layer_scale[layer_name] = scale_val;
            layer_zeropoint[layer_name] = zero_point;

            //            fprintf(stderr, "[%s] \tscale final %8.4f, zero point %8.4f\n", layer_name.c_str(), scale_val, zero_point);
        }
    }

    std::tr1::unordered_map<std::string, int> layer_used;
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* ir_node = ir_graph->node_list[i];
        for (int j = 0; j < ir_node->input_num; j++)
        {
            std::string layern = ir_graph->tensor_list[ir_node->input_tensors[j]]->name;
            layer_used[layern]++;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, optimize the calibration table.\n");
    /* process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip .... */
    if (inplace == 0)
    {
        for (int i = 0; i < ir_graph->tensor_num; i++)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                ir_tensor->scale = layer_scale[ir_tensor->name];
                ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
            }
        }
    }
    else
    {
        std::tr1::unordered_map<std::string, bool> layer_pass;
        for (int i = ir_graph->tensor_num - 1; i >= 0; i--)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                if (layer_pass[ir_tensor->name] == false)
                {
                    uint32_t ir_node_idx = ir_tensor->producer;
                    struct node* t_node = ir_graph->node_list[ir_node_idx];

                    std::string op_name = get_op_name_from_type(t_node->op.type);

                    bool poolTrue = false;
                    bool reluTrue = false;
                    if (op_name == "Pooling")
                    {
                        struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
                        if (pool_param->pool_method == 0)
                            poolTrue = true;
                    }
                    else if (op_name == "ReLU")
                    {
                        struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
                        if (relu_param->negative_slope == 0.f)
                            reluTrue = true;
                    }

                    if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" || op_name == "Slice" || poolTrue || reluTrue)
                    {
                        struct tensor* t_in_tensor = ir_graph->tensor_list[t_node->input_tensors[0]];
                        if (layer_scale[ir_tensor->name] != 0)
                        {
                            ir_tensor->scale = layer_scale[ir_tensor->name];
                            ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];

                            if (t_in_tensor->tensor_type == TENSOR_TYPE_VAR || t_in_tensor->tensor_type == TENSOR_TYPE_INPUT)
                            {
                                recursion_pass_through(ir_graph, ir_tensor->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                            }
                        }
                    }
                    else
                    {
                        ir_tensor->scale = layer_scale[ir_tensor->name];
                        ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
                    }
                    layer_pass[ir_tensor->name] = true;
                }
            }
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, quantize activation tensor done.\n");

    /* Set the params of acitvation ir_tensor */
    for (int i = 0; i < ir_graph->tensor_num; i++)
    {
        struct tensor* ir_tensor = ir_graph->tensor_list[i];
        if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
        {
            ir_tensor->data_type = TENGINE_DT_UINT8;
            ir_tensor->elem_size = sizeof(uint8_t);
        }
        ir_tensor->quant_param_num = 1;
    }

    /* Step 3 : set weight/bias quant scale value into ir_tensor, quant the weight params from Float32 to Int8 */
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* noden = ir_graph->node_list[i];
        std::string op_name = get_op_name_from_type(noden->op.type);

        /* quantize the tensor data from fp32 to uint8 */
        if (op_name == "Convolution" || op_name == "FullyConnected" || op_name == "Deconvolution")
        {
            /* Step 3.1 : quant weight */
            struct tensor* weight_tensor = ir_graph->tensor_list[noden->input_tensors[1]];

            if (weight_tensor->data_type == TENGINE_DT_FP32)
            {
                uint8_t* u8_weight_data = (uint8_t*)sys_malloc(weight_tensor->elem_num);
                float* weight_data = (float*)weight_tensor->data;

                /* calculate the quant scale value of weight perchannel, scale = (min-max / 255) */
                float weight_max = 0;
                float weight_min = 0;
                float weight_scale = 0;
                int weight_zero_point = 0;

                if (internal)
                {
                    weight_scale = weight_tensor->scale;
                    weight_zero_point = weight_tensor->zero_point;
                }
                else
                {
                    weight_max = *std::max_element(weight_data, weight_data + weight_tensor->elem_num);
                    weight_min = *std::min_element(weight_data, weight_data + weight_tensor->elem_num);
                    weight_scale = (weight_max - weight_min) / 255.f;
                    weight_zero_point = int(-weight_min / weight_scale);
                }
                //            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

                if (weight_scale > 10000000)
                    fprintf(stderr, "R U KIDDING ME??? %f\n", weight_scale);

                /* quantize the value of weight from Float32 to UInt8, value_u8 = (value_fp32 / scale).round().clip(0, 255) */
                float weight_data_tmp;
                for (int wi = 0; wi < weight_tensor->elem_num; wi++)
                {
                    //                weight_data[wi] = std::round(weight_data[wi] / weight_scale + static_cast<float>(weight_zero_point) );
                    //                weight_data[wi] = weight_data[wi] > 255 ? 255 : weight_data[wi];
                    //                weight_data[wi] = weight_data[wi] < 0 ? 0 : weight_data[wi];
                    weight_data_tmp = std::round(weight_data[wi] / weight_scale + static_cast<float>(weight_zero_point));
                    weight_data_tmp = weight_data_tmp > 255 ? 255 : weight_data_tmp;
                    weight_data_tmp = weight_data_tmp < 0 ? 0 : weight_data_tmp;
                    u8_weight_data[wi] = static_cast<uint8_t>(weight_data_tmp);
                }

                weight_tensor->scale = weight_scale;
                weight_tensor->zero_point = weight_zero_point;
                weight_tensor->data_type = TENGINE_DT_UINT8;
                weight_tensor->elem_size = sizeof(uint8_t);
                weight_tensor->data = u8_weight_data;

                /* step 3.2 : quant bias */
                if (noden->input_num > 2)
                {
                    struct tensor* input_tensor = ir_graph->tensor_list[noden->input_tensors[0]];
                    struct tensor* bias_tensor = ir_graph->tensor_list[noden->input_tensors[2]];

                    int* int32_bias_data = (int*)sys_malloc(bias_tensor->elem_num * bias_tensor->elem_size);
                    float* bias_data = (float*)bias_tensor->data;

                    /* calculate the quant scale value of bias perchannel, scale = scale_weight * scale_in */
                    float bias_scale = input_tensor->scale * weight_tensor->scale;

                    /* quantize the value of bias from Float32 to Int32, value_i32 = (value_fp32 / scale).round() */
                    for (int bi = 0; bi < bias_tensor->elem_num; bi++)
                    {
                        if (bias_scale == 0)
                            int32_bias_data[bi] = 0;
                        else
                        {
                            bias_data[bi] = roundf(bias_data[bi] / bias_scale);
                            int32_bias_data[bi] = int(bias_data[bi]);
                        }
                    }

                    bias_tensor->scale = bias_scale;
                    bias_tensor->data_type = TENGINE_DT_INT32;
                    bias_tensor->data = int32_bias_data;

                    //                fprintf(stderr, "[bias]   scale final %8.4f\n", bias_scale);
                }
            }
        }
        /* quantize the tensor data from fp32 to fp16, for TIM-VX NPU IP */
        else if (op_name == "PReLU")
        {
            for (int j = 0; j < noden->input_num; j++)
            {
                struct tensor* in_tensor = ir_graph->tensor_list[noden->input_tensors[j]];
                if (in_tensor->tensor_type == TENSOR_TYPE_CONST)
                {
                    float* fp32_data = (float*)in_tensor->data;
                    int data_elem = in_tensor->elem_num;

                    __fp16* fp16_data = (__fp16*)sys_malloc(data_elem * sizeof(__fp16));

                    for (int k = 0; k < data_elem; k++)
                    {
                        fp16_data[k] = fp32_to_fp16(fp32_data[k]);
                    }

                    in_tensor->data_type = TENGINE_DT_FP16;
                    in_tensor->data = fp16_data;
                    in_tensor->quant_param_num = 0;
                }
            }
        }
        else if (op_name == "Slice")
        {
            struct tensor* slice_input_tensor = get_ir_graph_tensor(ir_graph, noden->input_tensors[0]);
            struct tensor* slice_output_tensor = get_ir_graph_tensor(ir_graph, noden->output_tensors[0]);
            slice_output_tensor->scale = slice_input_tensor->scale;
            slice_output_tensor->zero_point = slice_input_tensor->zero_point;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 5, quantize weight tensor done.\n");

    if (!save_graph(ir_graph, output_file.c_str()))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 6, save UInt8 tmfile done, %s\n", output_file.c_str());

    return 0;
}

int save_graph_i8_perchannel(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal)
{
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again\n");

    /* Step 1 : create graph, load tengine model xxx.tmfile */
    struct graph* ir_graph = (struct graph*)create_graph(nullptr, "tengine", model_file);
    if (nullptr == ir_graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again done.\n");

    std::tr1::unordered_map<std::string, float> layer_scale;
    std::tr1::unordered_map<std::string, float> layer_zeropoint;

    fprintf(stderr, "[Quant Tools Info]: Step 3, load calibration table file %s.\n", scale_file);
    /* Step 2 : set activation quant scale value into ir_tensor */
    if (nullptr != scale_file)
    {
        std::ifstream scales(scale_file);
        std::string line;
        while (std::getline(scales, line))
        {
            std::string layer_name;
            float scale_val = 0.f;
            float zero_point = 0.f;
            size_t last = 0;
            size_t index = line.find_first_of(' ', last);
            size_t idx = line.find_last_of(' ', line.size());
            layer_name = line.substr(last, index - last);
            last = index + 1;
            scale_val = atof((line.substr(last, line.size() - last)).c_str());
            zero_point = atof((line.substr(idx + 1, line.size())).c_str());

            layer_scale[layer_name] = scale_val;
            layer_zeropoint[layer_name] = zero_point;

            //            fprintf(stderr, "[%s] \tscale final %8.4f, zero point %8.4f\n", layer_name.c_str(), scale_val, zero_point);
        }
    }

    std::tr1::unordered_map<std::string, int> layer_used;
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* ir_node = ir_graph->node_list[i];
        for (int j = 0; j < ir_node->input_num; j++)
        {
            std::string layern = ir_graph->tensor_list[ir_node->input_tensors[j]]->name;
            layer_used[layern]++;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, optimize the calibration table.\n");
    /* process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip .... */
    if (inplace == 0)
    {
        for (int i = 0; i < ir_graph->tensor_num; i++)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                ir_tensor->scale = layer_scale[ir_tensor->name];
                ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
            }
        }
    }
    else
    {
        std::tr1::unordered_map<std::string, bool> layer_pass;
        for (int i = ir_graph->tensor_num - 1; i >= 0; i--)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                if (layer_pass[ir_tensor->name] == false)
                {
                    uint32_t ir_node_idx = ir_tensor->producer;
                    struct node* t_node = ir_graph->node_list[ir_node_idx];

                    std::string op_name = get_op_name_from_type(t_node->op.type);

                    bool poolTrue = false;
                    bool reluTrue = false;
                    if (op_name == "Pooling")
                    {
                        struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
                        if (pool_param->pool_method == 0)
                            poolTrue = true;
                    }
                    else if (op_name == "ReLU")
                    {
                        struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
                        if (relu_param->negative_slope == 0.f)
                            reluTrue = true;
                    }

                    if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" || op_name == "Slice" || poolTrue || reluTrue)
                    {
                        struct tensor* t_in_tensor = ir_graph->tensor_list[t_node->input_tensors[0]];
                        if (layer_scale[ir_tensor->name] != 0)
                        {
                            ir_tensor->scale = layer_scale[ir_tensor->name];
                            ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];

                            if (t_in_tensor->tensor_type == TENSOR_TYPE_VAR || t_in_tensor->tensor_type == TENSOR_TYPE_INPUT)
                            {
                                recursion_pass_through(ir_graph, ir_tensor->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                            }
                        }
                    }
                    else
                    {
                        ir_tensor->scale = layer_scale[ir_tensor->name];
                        ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
                    }
                    layer_pass[ir_tensor->name] = true;
                }
            }
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, quantize activation tensor done.\n");

    /* Set the params of acitvation ir_tensor */
    for (int i = 0; i < ir_graph->tensor_num; i++)
    {
        struct tensor* ir_tensor = ir_graph->tensor_list[i];
        if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
        {
            ir_tensor->data_type = TENGINE_DT_INT8;
            ir_tensor->elem_size = sizeof(int8_t);
        }
        ir_tensor->quant_param_num = 1;
    }

    /* Step 3 : set weight/bias quant scale value into ir_tensor, quant the weight params from Float32 to Int8 */
    FILE* fp_weight = fopen("scale_weight.txt", "wb");
    FILE* fp_bias = fopen("scale_bias.txt", "wb");
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* noden = ir_graph->node_list[i];
        std::string op_name = get_op_name_from_type(noden->op.type);

        /* quantize the tensor data from fp32 to uint8 */
        if (op_name == "Convolution" || op_name == "FullyConnected" || op_name == "Deconvolution")
        {
            /* Step 3.1 : quant weight */
            struct tensor* weight_tensor = ir_graph->tensor_list[noden->input_tensors[1]];

            int channel_num = weight_tensor->dims[0];
            int cstep = int(weight_tensor->elem_num / channel_num);
            float* weight_data = (float*)weight_tensor->data;
            int8_t* i8_weight_data = (int8_t*)sys_malloc(weight_tensor->elem_num * sizeof(int8_t));

            float* weight_scale_list = (float*)sys_malloc(channel_num * sizeof(float));
            int* weight_zp_list = (int*)sys_malloc(channel_num * sizeof(int));

            fprintf(fp_weight, "%s ", weight_tensor->name);
            /* calculate the quant scale value of weight perchannel, scale = abs(min, max) / 127 */
            if (internal)
            {
                // TODO
                for (int ch = 0; ch < channel_num; ch++)
                {
                    weight_scale_list[ch] = weight_tensor->scale_list[ch];
                    weight_zp_list[ch] = 0;
                }
            }
            else
            {
                for (int ch = 0; ch < channel_num; ch++)
                {
                    float* weight_data_ch_start = weight_data + ch * cstep;
                    float* weight_data_ch_end = weight_data + (ch + 1) * cstep;
                    float weight_max = *std::max_element(weight_data_ch_start, weight_data_ch_end);
                    float weight_min = *std::min_element(weight_data_ch_start, weight_data_ch_end);

                    weight_scale_list[ch] = std::max(std::abs(weight_max), std::abs(weight_min)) / 127.f;
                    weight_zp_list[ch] = 0;
                    fprintf(fp_weight, "%8.8f ", weight_scale_list[ch]);
                }
                fprintf(fp_weight, "\n");
            }
            //            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

            /* quantize the value of weight from Float32 to Int8, value_i8 = (value_fp32 / scale).round().clip(-127, 127) */
            for (int ch = 0; ch < channel_num; ch++)
            {
                for (int j = 0; j < cstep; j++)
                {
                    if (weight_data[ch * cstep + j] == 0 || weight_scale_list[ch] == 0)
                        i8_weight_data[ch * cstep + j] = 0;
                    else
                    {
                        float int8_data = round(weight_data[ch * cstep + j] / weight_scale_list[ch]);
                        int8_data = int8_data > 127.f ? 127.f : int8_data;
                        int8_data = int8_data < -127.f ? -127.f : int8_data;
                        i8_weight_data[ch * cstep + j] = int8_t(int8_data);
                    }
                }
            }

            weight_tensor->scale_list = weight_scale_list;
            weight_tensor->zp_list = weight_zp_list;
            weight_tensor->data_type = TENGINE_DT_INT8;
            weight_tensor->elem_size = sizeof(int8_t); // int8, signed char
            weight_tensor->data = i8_weight_data;
            weight_tensor->quant_param_num = channel_num;

            /* step 3.2 : quant bias */
            if (noden->input_num > 2)
            {
                struct tensor* input_tensor = ir_graph->tensor_list[noden->input_tensors[0]];
                struct tensor* bias_tensor = ir_graph->tensor_list[noden->input_tensors[2]];

                float* bias_scale_list = (float*)sys_malloc(bias_tensor->dims[0] * sizeof(float));
                int* bias_zp_list = (int*)sys_malloc(bias_tensor->dims[0] * sizeof(int32_t));

                float* bias_data = (float*)bias_tensor->data;
                int* int32_bias_data = (int*)sys_malloc(bias_tensor->elem_num * sizeof(int32_t));

                int bstep = int(bias_tensor->elem_num / channel_num);

                fprintf(fp_bias, "%s ", bias_tensor->name);

                /* calculate the quant scale value of bias perchannel, scale = scale_weight * scale_in */
                for (int ch = 0; ch < channel_num; ch++)
                {
                    bias_scale_list[ch] = weight_scale_list[ch] * input_tensor->scale;
                    bias_zp_list[ch] = 0;

                    fprintf(fp_bias, "%8.8f ", bias_scale_list[ch]);
                }
                fprintf(fp_bias, "\n");

                /* quantize the value of bias from Float32 to Int32, value_i32 = (value_fp32 / scale).round() */
                for (int ch = 0; ch < channel_num; ch++)
                {
                    for (int bi = 0; bi < bstep; bi++)
                    {
                        if (bias_data[ch * bstep + bi] == 0 || bias_scale_list[ch] == 0)
                            int32_bias_data[ch * bstep + bi] = 0;
                        else
                            int32_bias_data[ch * bstep + bi] = int(round(bias_data[ch * bstep + bi] / bias_scale_list[ch]));
                    }
                }

                bias_tensor->scale_list = bias_scale_list;
                bias_tensor->zp_list = bias_zp_list;
                bias_tensor->data_type = TENGINE_DT_INT32;
                bias_tensor->elem_size = sizeof(int32_t); // int32, signed int
                bias_tensor->data = int32_bias_data;
                bias_tensor->quant_param_num = channel_num;

                // fprintf(stderr, "bias   %8.8f \t%s\n", bias_scale_list[0], bias_tensor->name);
            }
            // fprintf(stderr, "\n");
        }
        /* quantize the tensor data from fp32 to fp16, for TIM-VX NPU IP */
        else if (op_name == "PReLU")
        {
            for (int j = 0; j < noden->input_num; j++)
            {
                struct tensor* in_tensor = ir_graph->tensor_list[noden->input_tensors[j]];
                if (in_tensor->tensor_type == TENSOR_TYPE_CONST)
                {
                    float* fp32_data = (float*)in_tensor->data;
                    int data_elem = in_tensor->elem_num;

                    __fp16* fp16_data = (__fp16*)sys_malloc(data_elem * sizeof(__fp16));

                    for (int k = 0; k < data_elem; k++)
                    {
                        fp16_data[k] = fp32_to_fp16(fp32_data[k]);
                    }

                    in_tensor->data_type = TENGINE_DT_FP16;
                    in_tensor->data = fp16_data;
                    in_tensor->quant_param_num = 0;
                }
            }
        }
        else if (op_name == "Slice")
        {
            struct tensor* slice_input_tensor = get_ir_graph_tensor(ir_graph, noden->input_tensors[0]);
            struct tensor* slice_output_tensor = get_ir_graph_tensor(ir_graph, noden->output_tensors[0]);
            slice_output_tensor->scale = slice_input_tensor->scale;
            slice_output_tensor->zero_point = slice_input_tensor->zero_point;
        }
    }

    fclose(fp_weight);
    fclose(fp_bias);

    fprintf(stderr, "[Quant Tools Info]: Step 5, quantize weight tensor done.\n");

    if (!save_graph(ir_graph, output_file.c_str()))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 6, save Int8 tmfile done, %s\n", output_file.c_str());

    return 0;
}

int save_graph_u8_perchannel(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, bool internal)
{
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again\n");

    /* Step 1 : create graph, load tengine model xxx.tmfile */
    struct graph* ir_graph = (struct graph*)create_graph(nullptr, "tengine", model_file);
    if (nullptr == ir_graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again done.\n");

    std::tr1::unordered_map<std::string, float> layer_scale;
    std::tr1::unordered_map<std::string, float> layer_zeropoint;

    fprintf(stderr, "[Quant Tools Info]: Step 3, load calibration table file %s.\n", scale_file);
    /* Step 2 : set activation quant scale value into ir_tensor */
    if (nullptr != scale_file)
    {
        std::ifstream scales(scale_file);
        std::string line;
        while (std::getline(scales, line))
        {
            std::string layer_name;
            float scale_val = 0.f;
            float zero_point = 0.f;
            size_t last = 0;
            size_t index = line.find_first_of(' ', last);
            size_t idx = line.find_last_of(' ', line.size());
            layer_name = line.substr(last, index - last);
            last = index + 1;
            scale_val = atof((line.substr(last, line.size() - last)).c_str());
            zero_point = atof((line.substr(idx + 1, line.size())).c_str());

            layer_scale[layer_name] = scale_val;
            layer_zeropoint[layer_name] = zero_point;

            //            fprintf(stderr, "[%s] \tscale final %8.4f, zero point %8.4f\n", layer_name.c_str(), scale_val, zero_point);
        }
    }

    std::tr1::unordered_map<std::string, int> layer_used;
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* ir_node = ir_graph->node_list[i];
        for (int j = 0; j < ir_node->input_num; j++)
        {
            std::string layern = ir_graph->tensor_list[ir_node->input_tensors[j]]->name;
            layer_used[layern]++;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, optimize the calibration table.\n");
    /* process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip .... */
    if (inplace == 0)
    {
        for (int i = 0; i < ir_graph->tensor_num; i++)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                ir_tensor->scale = layer_scale[ir_tensor->name];
                ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
            }
        }
    }
    else
    {
        std::tr1::unordered_map<std::string, bool> layer_pass;
        for (int i = ir_graph->tensor_num - 1; i >= 0; i--)
        {
            struct tensor* ir_tensor = ir_graph->tensor_list[i];
            if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
            {
                if (layer_pass[ir_tensor->name] == false)
                {
                    uint32_t ir_node_idx = ir_tensor->producer;
                    struct node* t_node = ir_graph->node_list[ir_node_idx];

                    std::string op_name = get_op_name_from_type(t_node->op.type);

                    bool poolTrue = false;
                    bool reluTrue = false;
                    if (op_name == "Pooling")
                    {
                        struct pool_param* pool_param = (struct pool_param*)t_node->op.param_mem;
                        if (pool_param->pool_method == 0)
                            poolTrue = true;
                    }
                    else if (op_name == "ReLU")
                    {
                        struct relu_param* relu_param = (struct relu_param*)t_node->op.param_mem;
                        if (relu_param->negative_slope == 0.f)
                            reluTrue = true;
                    }

                    if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" || op_name == "Slice" || poolTrue || reluTrue)
                    {
                        struct tensor* t_in_tensor = ir_graph->tensor_list[t_node->input_tensors[0]];
                        if (layer_scale[ir_tensor->name] != 0)
                        {
                            ir_tensor->scale = layer_scale[ir_tensor->name];
                            ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];

                            if (t_in_tensor->tensor_type == TENSOR_TYPE_VAR || t_in_tensor->tensor_type == TENSOR_TYPE_INPUT)
                            {
                                recursion_pass_through(ir_graph, ir_tensor->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                            }
                        }
                    }
                    else
                    {
                        ir_tensor->scale = layer_scale[ir_tensor->name];
                        ir_tensor->zero_point = layer_zeropoint[ir_tensor->name];
                    }
                    layer_pass[ir_tensor->name] = true;
                }
            }
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, quantize activation tensor done.\n");

    /* Set the params of activation ir_tensor */
    for (int i = 0; i < ir_graph->tensor_num; i++)
    {
        struct tensor* ir_tensor = ir_graph->tensor_list[i];
        if (ir_tensor->tensor_type == TENSOR_TYPE_VAR || ir_tensor->tensor_type == TENSOR_TYPE_INPUT)
        {
            ir_tensor->data_type = TENGINE_DT_UINT8;
            ir_tensor->elem_size = sizeof(uint8_t);
        }
        ir_tensor->quant_param_num = 1;
    }

    /* Step 3 : set weight/bias quant scale value into ir_tensor, quant the weight params from Float32 to Int8 */
    FILE* fp_weight = fopen("scale_weight.txt", "wb");
    FILE* fp_bias = fopen("scale_bias.txt", "wb");
    for (int i = 0; i < ir_graph->node_num; i++)
    {
        struct node* noden = ir_graph->node_list[i];
        std::string op_name = get_op_name_from_type(noden->op.type);

        /* quantize the tensor data from fp32 to uint8 */
        if (op_name == "Convolution")
        {
            /* Step 3.1 : quant weight */
            struct tensor* weight_tensor = ir_graph->tensor_list[noden->input_tensors[1]];

            if (weight_tensor->dims[0] == 1 || noden->input_num == 2)
            {
                fprintf(stderr, "weight_tensor->name %s\n", weight_tensor->name);
                uint8_t* u8_weight_data = (uint8_t*)sys_malloc(weight_tensor->elem_num);
                float* weight_data = (float*)weight_tensor->data;

                /* calculate the quant scale value of weight perchannel, scale = (min-max / 255) */
                float weight_max = 0;
                float weight_min = 0;
                float weight_scale = 0;
                int weight_zero_point = 0;

                if (internal)
                {
                    weight_scale = weight_tensor->scale;
                    weight_zero_point = weight_tensor->zero_point;
                }
                else
                {
                    weight_max = *std::max_element(weight_data, weight_data + weight_tensor->elem_num);
                    weight_min = *std::min_element(weight_data, weight_data + weight_tensor->elem_num);
                    weight_scale = (weight_max - weight_min) / 255.f;
                    weight_zero_point = int(-weight_min / weight_scale);
                }
                //            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

                if (weight_scale > 10000000)
                    fprintf(stderr, "R U KIDDING ME??? %f\n", weight_scale);

                /* quantize the value of weight from Float32 to UInt8, value_u8 = (value_fp32 / scale).round().clip(0, 255) */
                float weight_data_tmp;
                for (int wi = 0; wi < weight_tensor->elem_num; wi++)
                {
                    //                weight_data[wi] = std::round(weight_data[wi] / weight_scale + static_cast<float>(weight_zero_point) );
                    //                weight_data[wi] = weight_data[wi] > 255 ? 255 : weight_data[wi];
                    //                weight_data[wi] = weight_data[wi] < 0 ? 0 : weight_data[wi];
                    weight_data_tmp = std::round(weight_data[wi] / weight_scale + static_cast<float>(weight_zero_point));
                    weight_data_tmp = weight_data_tmp > 255 ? 255 : weight_data_tmp;
                    weight_data_tmp = weight_data_tmp < 0 ? 0 : weight_data_tmp;
                    u8_weight_data[wi] = static_cast<uint8_t>(weight_data_tmp);
                }

                weight_tensor->scale = weight_scale;
                weight_tensor->zero_point = weight_zero_point;
                weight_tensor->data_type = TENGINE_DT_UINT8;
                weight_tensor->elem_size = sizeof(uint8_t);
                weight_tensor->data = u8_weight_data;
                weight_tensor->quant_param_num = 1;

                /* step 3.2 : quant bias */
                if (noden->input_num > 2)
                {
                    struct tensor* input_tensor = ir_graph->tensor_list[noden->input_tensors[0]];
                    struct tensor* bias_tensor = ir_graph->tensor_list[noden->input_tensors[2]];

                    int* int32_bias_data = (int*)sys_malloc(bias_tensor->elem_num * bias_tensor->elem_size);
                    float* bias_data = (float*)bias_tensor->data;

                    /* calculate the quant scale value of bias perchannel, scale = scale_weight * scale_in */
                    float bias_scale = input_tensor->scale * weight_tensor->scale;

                    /* quantize the value of bias from Float32 to Int32, value_i32 = (value_fp32 / scale).round() */
                    for (int bi = 0; bi < bias_tensor->elem_num; bi++)
                    {
                        if (bias_scale == 0)
                            int32_bias_data[bi] = 0;
                        else
                        {
                            bias_data[bi] = roundf(bias_data[bi] / bias_scale);
                            int32_bias_data[bi] = int(bias_data[bi]);
                        }
                    }

                    bias_tensor->scale = bias_scale;
                    bias_tensor->data_type = TENGINE_DT_INT32;
                    bias_tensor->data = int32_bias_data;
                    bias_tensor->quant_param_num = 1;

                    //                fprintf(stderr, "[bias]   scale final %8.4f\n", bias_scale);
                }
            }
            else
            {
                int channel_num = weight_tensor->dims[0];
                int cstep = int(weight_tensor->elem_num / channel_num);
                float* weight_data = (float*)weight_tensor->data;
                int8_t* i8_weight_data = (int8_t*)sys_malloc(weight_tensor->elem_num * sizeof(int8_t));

                float* weight_scale_list = (float*)sys_malloc(channel_num * sizeof(float));
                int* weight_zp_list = (int*)sys_malloc(channel_num * sizeof(int));

                fprintf(fp_weight, "%s ", weight_tensor->name);
                /* calculate the quant scale value of weight perchannel, scale = abs(min, max) / 127 */
                if (internal)
                {
                    // TODO
                }
                else
                {
                    for (int ch = 0; ch < channel_num; ch++)
                    {
                        float* weight_data_ch_start = weight_data + ch * cstep;
                        float* weight_data_ch_end = weight_data + (ch + 1) * cstep;
                        float weight_max = *std::max_element(weight_data_ch_start, weight_data_ch_end);
                        float weight_min = *std::min_element(weight_data_ch_start, weight_data_ch_end);

                        weight_scale_list[ch] = std::max(std::abs(weight_max), std::abs(weight_min)) / 127.f;
                        weight_zp_list[ch] = 0;
                        fprintf(fp_weight, "%8.8f ", weight_scale_list[ch]);
                    }
                    fprintf(fp_weight, "\n");
                }
                //            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

                /* quantize the value of weight from Float32 to Int8, value_i8 = (value_fp32 / scale).round().clip(-127, 127) */
                for (int ch = 0; ch < channel_num; ch++)
                {
                    for (int j = 0; j < cstep; j++)
                    {
                        if (weight_data[ch * cstep + j] == 0 || weight_scale_list[ch] == 0)
                            i8_weight_data[ch * cstep + j] = 0;
                        else
                        {
                            float int8_data = round(weight_data[ch * cstep + j] / weight_scale_list[ch]);
                            int8_data = int8_data > 127.f ? 127.f : int8_data;
                            int8_data = int8_data < -127.f ? -127.f : int8_data;
                            i8_weight_data[ch * cstep + j] = int8_t(int8_data);
                        }
                    }
                }

                weight_tensor->scale_list = weight_scale_list;
                weight_tensor->zp_list = weight_zp_list;
                weight_tensor->data_type = TENGINE_DT_INT8;
                weight_tensor->elem_size = sizeof(int8_t); // int8, signed char
                weight_tensor->data = i8_weight_data;
                weight_tensor->quant_param_num = channel_num;

                /* step 3.2 : quant bias */
                if (noden->input_num > 2)
                {
                    struct tensor* input_tensor = ir_graph->tensor_list[noden->input_tensors[0]];
                    struct tensor* bias_tensor = ir_graph->tensor_list[noden->input_tensors[2]];

                    float* bias_scale_list = (float*)sys_malloc(bias_tensor->dims[0] * sizeof(float));
                    int* bias_zp_list = (int*)sys_malloc(bias_tensor->dims[0] * sizeof(int32_t));

                    float* bias_data = (float*)bias_tensor->data;
                    int* int32_bias_data = (int*)sys_malloc(bias_tensor->elem_num * sizeof(int32_t));

                    int bstep = int(bias_tensor->elem_num / channel_num);

                    fprintf(fp_bias, "%s ", bias_tensor->name);

                    /* calculate the quant scale value of bias perchannel, scale = scale_weight * scale_in */
                    for (int ch = 0; ch < channel_num; ch++)
                    {
                        bias_scale_list[ch] = weight_scale_list[ch] * input_tensor->scale;
                        bias_zp_list[ch] = 0;

                        fprintf(fp_bias, "%8.8f ", bias_scale_list[ch]);
                    }
                    fprintf(fp_bias, "\n");

                    /* quantize the value of bias from Float32 to Int32, value_i32 = (value_fp32 / scale).round() */
                    for (int ch = 0; ch < channel_num; ch++)
                    {
                        for (int bi = 0; bi < bstep; bi++)
                        {
                            if (bias_data[ch * bstep + bi] == 0 || bias_scale_list[ch] == 0)
                                int32_bias_data[ch * bstep + bi] = 0;
                            else
                                int32_bias_data[ch * bstep + bi] = int(round(bias_data[ch * bstep + bi] / bias_scale_list[ch]));
                        }
                    }

                    bias_tensor->scale_list = bias_scale_list;
                    bias_tensor->zp_list = bias_zp_list;
                    bias_tensor->data_type = TENGINE_DT_INT32;
                    bias_tensor->elem_size = sizeof(int32_t); // int32, signed int
                    bias_tensor->data = int32_bias_data;
                    bias_tensor->quant_param_num = channel_num;
                }

                // fprintf(stderr, "bias   %8.8f \t%s\n", bias_scale_list[0], bias_tensor->name);
            }
            // fprintf(stderr, "\n");
        }
        else if (op_name == "FullyConnected" || op_name == "Deconvolution")
        {
            /* Step 3.1 : quant weight */
            struct tensor* weight_tensor = ir_graph->tensor_list[noden->input_tensors[1]];

            uint8_t* u8_weight_data = (uint8_t*)sys_malloc(weight_tensor->elem_num * sizeof(uint8_t));
            float* weight_data = (float*)weight_tensor->data;

            /* calculate the quant scale value of weight perchannel, scale = (min-max / 255) */
            float weight_max = 0;
            float weight_min = 0;
            float weight_scale = 0;
            int weight_zero_point = 0;

            if (internal)
            {
                weight_scale = weight_tensor->scale;
                weight_zero_point = weight_tensor->zero_point;
            }
            else
            {
                weight_max = *std::max_element(weight_data, weight_data + weight_tensor->elem_num);
                weight_min = *std::min_element(weight_data, weight_data + weight_tensor->elem_num);
                weight_scale = (weight_max - weight_min) / 255.f;
                weight_zero_point = int(-weight_min / weight_scale);
            }
            //            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

            /* quantize the value of weight from Float32 to UInt8, value_u8 = (value_fp32 / scale).round().clip(0, 255) */
            for (int wi = 0; wi < weight_tensor->elem_num; wi++)
            {
                weight_data[wi] = roundf(weight_data[wi] / weight_scale + (float)weight_zero_point);
                weight_data[wi] = weight_data[wi] > 255.f ? 255.f : weight_data[wi];
                weight_data[wi] = weight_data[wi] < 0.f ? 0.f : weight_data[wi];
                u8_weight_data[wi] = uint8_t(weight_data[wi]);
            }

            weight_tensor->scale = weight_scale;
            weight_tensor->zero_point = weight_zero_point;
            weight_tensor->data_type = TENGINE_DT_UINT8;
            weight_tensor->elem_size = sizeof(uint8_t);
            weight_tensor->data = u8_weight_data;

            /* step 3.2 : quant bias */
            if (noden->input_num > 2)
            {
                struct tensor* input_tensor = ir_graph->tensor_list[noden->input_tensors[0]];
                struct tensor* bias_tensor = ir_graph->tensor_list[noden->input_tensors[2]];

                int* int32_bias_data = (int*)sys_malloc(bias_tensor->elem_num * bias_tensor->elem_size);
                float* bias_data = (float*)bias_tensor->data;

                /* calculate the quant scale value of bias perchannel, scale = scale_weight * scale_in */
                float bias_scale = input_tensor->scale * weight_tensor->scale;

                /* quantize the value of bias from Float32 to Int32, value_i32 = (value_fp32 / scale).round() */
                for (int bi = 0; bi < bias_tensor->elem_num; bi++)
                {
                    if (bias_scale == 0)
                        int32_bias_data[bi] = 0;
                    else
                    {
                        bias_data[bi] = roundf(bias_data[bi] / bias_scale);
                        int32_bias_data[bi] = int(bias_data[bi]);
                    }
                }

                bias_tensor->scale = bias_scale;
                bias_tensor->data_type = TENGINE_DT_INT32;
                bias_tensor->data = int32_bias_data;

                //                fprintf(stderr, "[bias]   scale final %8.4f\n", bias_scale);
            }
        }
        /* quantize the tensor data from fp32 to fp16, for TIM-VX NPU IP */
        else if (op_name == "PReLU")
        {
            for (int j = 0; j < noden->input_num; j++)
            {
                struct tensor* in_tensor = ir_graph->tensor_list[noden->input_tensors[j]];
                if (in_tensor->tensor_type == TENSOR_TYPE_CONST)
                {
                    float* fp32_data = (float*)in_tensor->data;
                    int data_elem = in_tensor->elem_num;

                    __fp16* fp16_data = (__fp16*)sys_malloc(data_elem * sizeof(__fp16));

                    for (int k = 0; k < data_elem; k++)
                    {
                        fp16_data[k] = fp32_to_fp16(fp32_data[k]);
                    }

                    in_tensor->data_type = TENGINE_DT_FP16;
                    in_tensor->data = fp16_data;
                    in_tensor->quant_param_num = 0;
                }
            }
        }
        else if (op_name == "Slice")
        {
            struct tensor* slice_input_tensor = get_ir_graph_tensor(ir_graph, noden->input_tensors[0]);
            struct tensor* slice_output_tensor = get_ir_graph_tensor(ir_graph, noden->output_tensors[0]);
            slice_output_tensor->scale = slice_input_tensor->scale;
            slice_output_tensor->zero_point = slice_input_tensor->zero_point;
        }
    }

    fclose(fp_weight);
    fclose(fp_bias);

    fprintf(stderr, "[Quant Tools Info]: Step 5, quantize weight tensor done.\n");

    if (!save_graph(ir_graph, output_file.c_str()))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 6, save Int8 tmfile done, %s\n", output_file.c_str());

    return 0;
}
