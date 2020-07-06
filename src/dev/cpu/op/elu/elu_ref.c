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
 * Author: zpluo@openailab.com
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "elu_param.h"

typedef struct __elu_param
{
    float scale;
    int zero_point;
    float alpha;
} _elu_param, *p_elu_param;

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

int ref_elu_fp32(float* data, float* out_data, int size, p_elu_param param)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
        {
            out_data[i] = (exp(data[i]) - 1) * param->alpha;
        }
        else
        {
            out_data[i] = data[i];
        }
    }
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct elu_param* param = ( struct elu_param* )node->op.param_mem;

    int elem_num = input_tensor->elem_num;
    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;

    float scale = 1.f;
    int zero_point = 0;

    _elu_param op_param;
    op_param.alpha = param->alpha;
    op_param.scale = scale;
    op_param.zero_point = zero_point;

    return ref_elu_fp32(in_data, out_data, elem_num, &op_param);
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
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
    return register_builtin_node_ops(OP_ELU, &hcl_node_ops);
}

static int unreg_relu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_ELU, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_relu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_relu_hcl_ops);
