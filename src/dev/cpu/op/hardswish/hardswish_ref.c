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
 * Author: renzun@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include <math.h>
#include "hardswish_param.h"

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
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct hardswish_param* hardswish_param = ( struct hardswish_param* )ir_node->op.param_mem;

    float alpha = hardswish_param->alpha;
    float beta = hardswish_param->beta;
    float lower = -beta / alpha;
    float upper = (1.f / alpha) + lower;

    int chan_num = (input_tensor->dims[0]) * (input_tensor->dims[1]);
    int chan_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);

    float* pdata = ( float* )input_tensor->data;
    float* pout_data = ( float* )output_tensor->data;
    int num_thread = exec_graph->num_thread;

#pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < chan_num; j++)
    {
        float* data = pdata + j * chan_size;
        float* out_data = pout_data + j * chan_size;

        for (int i = 0; i < chan_size; i++)
        {
            if (data[i] < lower)
                out_data[i] = 0.f;
            else if (data[i] > upper)
                out_data[i] = data[i];
            else
                out_data[i] = data[i] * (data[i] * alpha + beta);
        }
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    struct ir_node* ir_node = exec_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->data_type != TENGINE_DT_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
        return 0;

    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_hardswish_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_HARDSWISH, &hcl_node_ops);
}

static int unreg_hardswish_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_HARDSWISH, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_hardswish_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_hardswish_hcl_ops);
