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

#include "quant_save_graph.hpp"
#include "compiler_fp16.h"

void recursion_pass_through(struct graph* graphn, const char* layer_name, struct tensor* t,
                            std::tr1::unordered_map<std::string, int> &layer_used, std::tr1::unordered_map<std::string,float> &layer_scale,
                            std::tr1::unordered_map<std::string,float> &layer_zeropoint, std::tr1::unordered_map<std::string, int> &layer_pass)
{
    if (layer_pass[t->name] == 0 && layer_used[t->name] < 2)
    {
        t->scale = layer_scale[layer_name];
        t->zero_point = layer_zeropoint[layer_name];
        layer_scale[t->name] = layer_scale[layer_name];
        layer_zeropoint[t->name] = layer_zeropoint[layer_name];

        uint32_t ir_node_idx = t->producer;
        struct node* t_node = graphn->node_list[ir_node_idx];

        std::string op_name = get_op_name_from_type(t_node->op.type);
        bool poolTrue = false;
        bool reluTrue = false;
        if (op_name == "Pooling")
        {
            struct pool_param* pool_param = ( struct pool_param* )t_node->op.param_mem;
            if (pool_param->pool_method == 0)
                poolTrue = true;
        }
        else if (op_name == "ReLU")
        {
            struct relu_param* relu_param = ( struct relu_param* )t_node->op.param_mem;
            if (relu_param->negative_slope == 0.f)
                reluTrue = true;
        }
        if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" ||
            poolTrue || reluTrue)
        {
            struct tensor* t_in_tensor = graphn->tensor_list[t_node->input_tensors[0]];
            if (layer_scale[t->name] != 0)
            {
                if (t_in_tensor->tensor_type == 1 || t_in_tensor->tensor_type == 3)
                {
                    recursion_pass_through(graphn, t->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                }
            }
        }
        layer_pass[t->name] = 1;
    }
}

int save_graph_u8_perlayer(const char* model_file, const char* scale_file, const std::string& output_file, int inplace, int num_thread, bool internal)
{
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again\n");

    /* Step 1 : create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }
    fprintf(stderr, "[Quant Tools Info]: Step 3, load FP32 tmfile once again done.\n");

    std::tr1::unordered_map<std::string,float> layer_scale;
    std::tr1::unordered_map<std::string,float> layer_zeropoint;
    bool parse_from_file = false;

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
            size_t index = line.find_first_of(" ", last);
            size_t idx = line.find_last_of(" ", line.size());
            layer_name = line.substr(last, index - last);
            last = index + 1;
            scale_val = atof((line.substr(last, line.size() - last)).c_str());
            zero_point = atof((line.substr(idx + 1, line.size())).c_str());

            layer_scale[layer_name] = scale_val;
            layer_zeropoint[layer_name] = zero_point;

//            fprintf(stderr, "[%s] \tscale final %8.4f, zero point %8.4f\n", layer_name.c_str(), scale_val, zero_point);
        }
    }

    struct graph* graphn = (struct graph*)graph;

    std::tr1::unordered_map<std::string,int> layer_used;
    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* noden = graphn->node_list[i];
        for (int j = 0; j < noden->input_num; j++ )
        {
            std::string layern = graphn->tensor_list[noden->input_tensors[j]]->name;
            layer_used[layern] ++;
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, optimize the calibration table.\n");
    /* process the inplace quant scale of activation in some types of op, such as max pooling, ReLU, Flatten, Reshape, Clip .... */
    if (inplace == 0)
    {
        for (int i = 0; i < graphn->tensor_num; i++)
        {
            struct tensor* t = graphn->tensor_list[i];
            if (t->tensor_type == TENSOR_TYPE_VAR || t->tensor_type == TENSOR_TYPE_INPUT)
            {
                t->scale = layer_scale[t->name];
                t->zero_point = layer_zeropoint[t->name];
            }
        }
    }
    else
    {
        std::tr1::unordered_map<std::string, int> layer_pass;
        for (int i = graphn->tensor_num-1; i >= 0; i--)
        {
            struct tensor* t = graphn->tensor_list[i];
            if (t->tensor_type == TENSOR_TYPE_VAR || t->tensor_type == TENSOR_TYPE_INPUT)
            {
                if (layer_pass[t->name] == 0)
                {
                    uint32_t ir_node_idx = t->producer;
                    struct node* t_node = graphn->node_list[ir_node_idx];

                    std::string op_name = get_op_name_from_type(t_node->op.type);

                    bool poolTrue = false;
                    bool reluTrue = false;
                    if (op_name == "Pooling")
                    {
                        struct pool_param* pool_param = ( struct pool_param* )t_node->op.param_mem;
                        if (pool_param->pool_method == 0)
                            poolTrue = true;
                    }
                    else if (op_name == "ReLU")
                    {
                        struct relu_param* relu_param = ( struct relu_param* )t_node->op.param_mem;
                        if (relu_param->negative_slope == 0.f)
                            reluTrue = true;
                    }

                    if (op_name == "Flatten" || op_name == "Reshape" || op_name == "Squeeze" || op_name == "Clip" ||
                        poolTrue || reluTrue)
                    {
                        struct tensor* t_in_tensor = graphn->tensor_list[t_node->input_tensors[0]];
                        if (layer_scale[t->name] != 0)
                        {
                            t->scale = layer_scale[t->name];
                            t->zero_point = layer_zeropoint[t->name];

                            if (t_in_tensor->tensor_type == TENSOR_TYPE_VAR || t_in_tensor->tensor_type == TENSOR_TYPE_INPUT)
                            {
                                recursion_pass_through(graphn, t->name, t_in_tensor, layer_used, layer_scale, layer_zeropoint, layer_pass);
                            }
                        }
                    }
                    else
                    {
                        t->scale = layer_scale[t->name];
                        t->zero_point = layer_zeropoint[t->name];
                    }
                    layer_pass[t->name] = 1;
                }
            }
        }
    }

    fprintf(stderr, "[Quant Tools Info]: Step 4, quantize activation tensor done.\n");

    /* Set the params of acitvation ir_tensor */
    for (int i = 0; i < graphn->tensor_num; i++)
    {
        struct tensor* t = graphn->tensor_list[i];
        if (t->tensor_type == TENSOR_TYPE_VAR || t->tensor_type == TENSOR_TYPE_INPUT)
        {
            t->data_type = TENGINE_DT_UINT8;
            t->elem_size = sizeof(uint8_t);
        }
        t->quant_param_num = 1;
    }

    /* Step 3 : set weight/bias quant scale value into ir_tensor, quant the weight params from Float32 to Int8 */
    for (int i = 0; i < graphn->node_num; i++)
    {
        struct node* noden = graphn->node_list[i];
        std::string op_name = get_op_name_from_type(noden->op.type);

        /* quantize the tensor data from fp32 to uint8 */
        if (op_name == "Convolution" || op_name == "FullyConnected")
        {
            /* Step 3.1 : quant weight */
            struct tensor* weight_tensor = graphn->tensor_list[noden->input_tensors[1]];

            uint8_t * u8_weight_data = (uint8_t*)sys_malloc(weight_tensor->elem_num * sizeof(uint8_t));
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
                weight_zero_point = int(-weight_min/weight_scale);
            }
//            fprintf(stderr, "[weight] scale final %8.4f, zero point %4d\n", weight_scale, weight_zero_point);

            /* quantize the value of weight from Float32 to UInt8, value_u8 = (value_fp32 / scale).round().clip(0, 255) */
            for (int wi = 0; wi < weight_tensor->elem_num; wi++)
            {
                weight_data[wi] = round(weight_data[wi] / weight_scale + (float )weight_zero_point);
                weight_data[wi] = weight_data[wi] > 255.f ? 255.f : weight_data[wi];
                weight_data[wi] = weight_data[wi] < 0.f   ?   0.f : weight_data[wi];
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
                struct tensor* input_tensor = graphn->tensor_list[noden->input_tensors[0]];
                struct tensor* bias_tensor = graphn->tensor_list[noden->input_tensors[2]];

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
                        bias_data[bi] = round(bias_data[bi] / bias_scale);
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
                struct tensor* in_tensor = graphn->tensor_list[noden->input_tensors[j]];
                if (in_tensor->tensor_type == TENSOR_TYPE_CONST)
                {
                    float* fp32_data =  (float*) in_tensor->data;
                    int data_elem =  in_tensor->elem_num;

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
    }

    fprintf(stderr, "[Quant Tools Info]: Step 5, quantize weight tensor done.\n");

    if (!save_graph(graph, output_file.c_str()))
    {
        fprintf(stderr, "save graph failed.\n");
        return -1;
    }

    fprintf(stderr, "[Quant Tools Info]: Step 6, save Int8 tmfile done, %s\n", output_file.c_str());

    return 0;
}
