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
#include "upsample_param.h"
#include <math.h>

static int ref_upsample_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                             struct upsample_param* param, int num_thread)
// static int ref_upsample_fp32(float* input, float* output, upsample_param* param)
{
    float* input = input_tensor->data;
    float* output = output_tensor->data;

    float scale = param->scale;
    int batch = output_tensor->dims[0];
    int channel = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];

    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {
                    int in_w = w / scale;
                    int in_h = h / scale;
                    int out_idx = n * channel * out_h * out_w + c * out_h * out_w + h * out_w + w;
                    int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
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
    struct ir_tensor* roi_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct upsample_param* upsample_param = ( struct upsample_param* )ir_node->op.param_mem;

    ref_upsample_fp32(input_tensor, output_tensor, upsample_param, exec_graph->num_thread);

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

static int reg_upsample_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_UPSAMPLE, &hcl_node_ops);
}

static int unreg_upsample_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_UPSAMPLE, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_upsample_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_upsample_hcl_ops);
