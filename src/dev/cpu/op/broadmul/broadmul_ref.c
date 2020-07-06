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
 * Author: jjzeng@openailab.com
 */
#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

typedef struct __ref_broadmul_param
{
    int out_size;
    int on_size;
    int in_size;
    float in0_scale;
    float in1_scale;
    int in0_zero;
    int in1_zero;
} ref_broadmul_param, *p_ref_broadmul_param;

static int ref_broadmul_fp32(float* in0, float* in1, float* out, p_ref_broadmul_param param)
{
    int out_size = param->out_size;
    int in_size = param->in_size;
    int on_size = param->on_size;

    for (int o = 0; o < out_size; o++)
    {
        for (int j = 0; j < on_size; j++)
        {
            float data1 = in1[j];
            for (int i = 0; i < in_size; i++)
            {
                int index = (o * on_size + j) * in_size + i;
                out[index] = in0[index] * data1;
            }
        }
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;

    struct ir_tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);

    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int in_size = 1;
    int on_size = 1;
    int out_size = 1;

    int axis = 0;
    for (int ii = 0; ii < input1->dim_num; ++ii)
    {
        if (input1->dims[ii] == 1)
        {
            out_size = out_size * input0->dims[ii];
        }
        else
        {
            axis = 1;
            break;
        }
    }
    on_size = input0->dims[axis];
    for (int ii = axis + 1; ii < input0->dim_num; ++ii)
    {
        in_size = in_size * input0->dims[ii];
    }

    ref_broadmul_param param;
    param.in_size = in_size;
    param.out_size = out_size;
    param.on_size = on_size;

    return ref_broadmul_fp32(input0->data, input1->data, output->data, &param);
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

static int reg_relu_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_BROADMUL, &hcl_node_ops);
}

static int unreg_relu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_BROADMUL, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_relu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_relu_hcl_ops);
