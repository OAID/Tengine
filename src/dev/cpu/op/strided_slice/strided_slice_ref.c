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
 * Author: sqfu@openailab.com
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "strided_slice_param.h"

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
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct strided_slice_param* param = ( struct strided_slice_param* )ir_node->op.param_mem;

    int batch_num = input_tensor->dims[0];
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    int out_c = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];

    int out_chw = out_c * out_h * out_w;
    int out_hw = out_h * out_w;
    int in_chw = in_c * in_h * in_w;
    int in_hw = in_h * in_w;

    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    for (int n = 0; n < batch_num; n++)
    {
        for (int c = 0; c < out_c; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {
                    int input_index = (param->begin[0] + n * param->stride[0]) * in_chw +
                                      (param->begin[1] + c * param->stride[1]) * in_hw +
                                      (param->begin[2] + h * param->stride[2]) * in_w +
                                      (param->begin[3] + w * param->stride[3]);
                    int output_index = n * out_chw + c * out_hw + h * out_w + w;

                    output_data[output_index] = input_data[input_index];
                }
            }
        }
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops strided_slice_node_ops = {.prerun = NULL,
                                                 .run = run,
                                                 .reshape = NULL,
                                                 .postrun = NULL,
                                                 .init_node = init_node,
                                                 .release_node = release_node,
                                                 .score = score};

static int reg_strided_slice_ops(void* arg)
{
    return register_builtin_node_ops(OP_STRIDED_SLICE, &strided_slice_node_ops);
}

static int unreg_strided_slice_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_STRIDED_SLICE, &strided_slice_node_ops);
}

AUTO_REGISTER_OPS(reg_strided_slice_ops);
AUTO_UNREGISTER_OPS(unreg_strided_slice_ops);
