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
 * Author: bhu@openailab.com
 */

#include "relu_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <math.h>


static int ref_relu_fp32(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope)
{
    int total_size = input_tensor->elem_num;
    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    if (negative_slope == 0)
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_data[i] < 0)
                output_data[i] = 0;
            else
                output_data[i] = input_data[i];
        }
    }
    else
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_data[i] < 0)
                output_data[i] = input_data[i] * negative_slope;
            else
                output_data[i] = input_data[i];
        }
    }

    return 0;
}
#if MACOS
#else
static int ref_relu_fp16(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope,
                         int num_thread)
{
    int total_size = input_tensor->elem_num;
    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    /* cost fp16 to fp32 */
    fp16_t* input_fp16 = input_tensor->data;
    fp16_t* output_fp16 = output_tensor->data;
    float* input_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i< total_size; i++)
    {
        input_fp32[i] = fp16_to_fp32(input_fp16[i]);
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_fp32[i] < 0)
                input_fp32[i] = 0;
            else
                input_fp32[i] = input_fp32[i];
        }
    }
    else
    {
        for (int i = 0; i < total_size; i++)
        {
            if (input_fp32[i] < 0)
                input_fp32[i] = input_fp32[i] * negative_slope;
            else
                input_fp32[i] = input_fp32[i];
        }
    }

    /* cost fp32 to fp16 */
    for(int i=0; i<total_size; i++)
    {
        output_fp16[i] = fp32_to_fp16(input_fp32[i]);
    }

    sys_free(input_fp32);

    return 0;
}
#endif
static int ref_relu_uint8(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope,
                          int num_thread)
{
    int total_size = input_tensor->elem_num;

    /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i<total_size; i++)
    {
        data_fp32[i] = ((float )input_uint8[i] - (float )input_zero) * input_scale;
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int i = 0; i < total_size; i++)
        {
            if (data_fp32[i] < 0)
                data_fp32[i] = 0;
            else
                data_fp32[i] = data_fp32[i];
        }
    }
    else
    {
        for (int i = 0; i < total_size; i++)
        {
            if (data_fp32[i] < 0)
                data_fp32[i] = data_fp32[i] * negative_slope;
            else
                data_fp32[i] = data_fp32[i];
        }
    }

    /* quant */
    for(int i=0; i<total_size; i++)
    {
        int udata = round(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(data_fp32);

    return 0;
}

static int ref_relu_int8(struct tensor* input_tensor, struct tensor* output_tensor, float negative_slope)
{
    int total_size = input_tensor->elem_num;

    /* dequant */
    int8_t* input_int8 = input_tensor->data;
    int8_t* output_int8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;

    float* data_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for(int i=0; i<total_size; i++)
    {
        data_fp32[i] = (float )input_int8[i] * input_scale;
    }

    /* process */
    if (negative_slope == 0)
    {
        for (int i = 0; i < total_size; i++)
        {
            if (data_fp32[i] < 0)
                data_fp32[i] = 0;
            else
                data_fp32[i] = data_fp32[i];
        }
    }
    else
    {
        for (int i = 0; i < total_size; i++)
        {
            if (data_fp32[i] < 0)
                data_fp32[i] = data_fp32[i] * negative_slope;
            else
                data_fp32[i] = data_fp32[i];
        }
    }

    /* quant */
    for(int i=0; i<total_size; i++)
    {
        int data_i32 = round(data_fp32[i] / output_scale);
        if (data_i32 > 127)
            data_i32 = 127;
        else if (data_i32 < -127)
            data_i32 = -127;
        output_int8[i] = (int8_t)data_i32;
    }

    sys_free(data_fp32);

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct relu_param* relu_param = ( struct relu_param* )ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_relu_fp32(input_tensor, output_tensor, relu_param->negative_slope);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
        #if MACOS
        TLOG_ERR("FP16 not support mac os");
        #else
        ret = ref_relu_fp16(input_tensor, output_tensor, relu_param->negative_slope, exec_graph->num_thread);
        #endif
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_relu_uint8(input_tensor, output_tensor, relu_param->negative_slope, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_relu_int8(input_tensor, output_tensor, relu_param->negative_slope);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    int ret = set_ir_tensor_shape(output, input->dims, input->dim_num);
    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

int register_relu_ref_op()
{
    return register_builtin_node_ops(OP_RELU, &hcl_node_ops);
}

int unregister_relu_ref_op()
{
    return unregister_builtin_node_ops(OP_RELU, &hcl_node_ops);
}
