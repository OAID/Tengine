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
#include <math.h>

int ref_squareddifference_fp32(struct ir_tensor* input_tensor_0, struct ir_tensor* input_tensor_1,
                               struct ir_tensor* output_tensor, int num_thread)
{
    // dims size = 2 or 3
    if (input_tensor_0->dim_num < 4)
    {
        float* input0 = input_tensor_0->data;
        float* input1 = input_tensor_1->data;
        float* output = output_tensor->data;
        int total_size = output_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            output[i] = powf((input0[i] - input1[i]), 2);
        }

        return 0;
    }
    // dims size 3
    else if (output_tensor->dim_num == 4)
    {
        int w = output_tensor->dims[3];
        int h = output_tensor->dims[2];
        int channels = output_tensor->dims[1];
        int size = h * w;
        int c_step = h * w;

        float* input0 = input_tensor_0->data;
        float* input1 = input_tensor_1->data;
        float* output = output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src0 = input0 + c_step * q;
            float* src1 = input1 + c_step * q;
            float* dst = output + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = powf((src0[i] - src1[i]), 2);
            }
        }

        return 0;
    }

    return -1;
}

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor_0;
    struct ir_tensor* input_tensor_1;
    struct ir_tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor_0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    input_tensor_1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = ref_squareddifference_fp32(input_tensor_0, input_tensor_1, output_tensor, exec_graph->num_thread);
    if (ret != 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_squareddifference_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_SQUAREDDIFFERENCE, &hcl_node_ops);
}

static int unreg_squareddifference_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_SQUAREDDIFFERENCE, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_squareddifference_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_squareddifference_hcl_ops);
