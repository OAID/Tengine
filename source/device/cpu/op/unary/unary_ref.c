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
 * Author: qtang@openailab.com
 */

#include "unary_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>


static int ref_unary_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct unary_param* param,
                          int num_thread)
{
    float* in_data = input_tensor->data;
    float* out_data = output_tensor->data;

    int size = input_tensor->elem_num;

    int type = param->type;

    switch (type)
    {
        case 0:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = fabs(in_data[i]);
            }
            break;
        case 1:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = -(in_data[i]);
            }
            break;
        case 2:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = floor(in_data[i]);
            }
            break;
        case 3:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = ceil(in_data[i]);
            }
            break;
        case 4:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = in_data[i] * in_data[i];
            }
            break;
        case 5:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sqrt(in_data[i]);
            }
            break;
        case 6:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / sqrt(in_data[i]);
            }
            break;
        case 7:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = exp(in_data[i]);
            }
            break;
        case 8:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = log(in_data[i]);
            }
            break;
        case 9:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sin(in_data[i]);
            }
            break;
        case 10:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = cos(in_data[i]);
            }
            break;
        case 11:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tan(in_data[i]);
            }
            break;
        case 12:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = asin(in_data[i]);
            }
            break;
        case 13:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = acos(in_data[i]);
            }
            break;
        case 14:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = atan(in_data[i]);
            }
            break;
        case 15:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / (in_data[i]);
            }
            break;
        case 16:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tanh(in_data[i]);
            }
            break;
        default:
            break;
    }

    return 0;
}

static int ref_unary_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct unary_param* param,
                          int num_thread)
{
    /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* in_data = ( float* )sys_malloc(input_size * sizeof(float));
    float* out_data = ( float* )sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        in_data[i] = (( float )input_uint8[i] - ( float )input_zero) * input_scale;
    }

    int size = input_tensor->elem_num;

    int type = param->type;

    switch (type)
    {
        case 0:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = fabs(in_data[i]);
            }
            break;
        case 1:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = -(in_data[i]);
            }
            break;
        case 2:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = floor(in_data[i]);
            }
            break;
        case 3:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = ceil(in_data[i]);
            }
            break;
        case 4:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = in_data[i] * in_data[i];
            }
            break;
        case 5:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sqrt(in_data[i]);
            }
            break;
        case 6:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / sqrt(in_data[i]);
            }
            break;
        case 7:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = exp(in_data[i]);
            }
            break;
        case 8:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = log(in_data[i]);
            }
            break;
        case 9:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = sin(in_data[i]);
            }
            break;
        case 10:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = cos(in_data[i]);
            }
            break;
        case 11:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tan(in_data[i]);
            }
            break;
        case 12:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = asin(in_data[i]);
            }
            break;
        case 13:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = acos(in_data[i]);
            }
            break;
        case 14:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = atan(in_data[i]);
            }
            break;
        case 15:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = 1.f / (in_data[i]);
            }
            break;
        case 16:
            for (int i = 0; i < size; i++)
            {
                out_data[i] = tanh(in_data[i]);
            }
            break;
        default:
            break;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(in_data);
    sys_free(out_data);

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
    struct unary_param* unary_param = ( struct unary_param* )ir_node->op.param_mem;

	int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_unary_fp32(input_tensor, output_tensor, unary_param, exec_graph->num_thread);
    else if(input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_unary_uint8(input_tensor, output_tensor, unary_param, exec_graph->num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

int register_unary_ref_op()
{
    return register_builtin_node_ops(OP_UNARY, &hcl_node_ops);
}

int unregister_unary_ref_op()
{
    return unregister_builtin_node_ops(OP_UNARY, &hcl_node_ops);
}
