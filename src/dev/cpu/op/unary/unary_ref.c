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
 * Author: qtang@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "unary_param.h"
#include <math.h>

static int ref_unary_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct unary_param* param,
                          int num_thread)
{
    float* in_data = input_tensor->data;
    float* out_data = output_tensor->data;

    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int size = n * c * h * w;

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
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct unary_param* unary_param = ( struct unary_param* )ir_node->op.param_mem;

    ref_unary_fp32(input_tensor, output_tensor, unary_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
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

static int reg_unary_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_UNARY, &hcl_node_ops);
}

static int unreg_unary_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_UNARY, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_unary_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_unary_hcl_ops);
