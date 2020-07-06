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
 * Author: jxyang@openailab.com
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "gather_param.h"

typedef struct
{
    int in_shape[4];    // the dim of the input
    int axis;
    int indices_num;
    int dim_size;
} gather_param_t;

static int ref_gather_fp32(float* input, int* input_indices, float* output, gather_param_t* param, int num_thread)
{
    float* out_ptr = output;
    float* in_ptr = input;
    int axis = param->axis;
    int outer_size = 1;
    int inner_size = 1;
    int axis_size = param->in_shape[axis];

    for (int i = 0; i < axis; i++)
    {
        outer_size *= param->in_shape[i];
    }

    for (int i = axis + 1; i < param->dim_size; i++)
    {
        inner_size *= param->in_shape[i];
    }

	// #pragma omp parallel for num_threads(num_thread)
    for (int outer = 0; outer < outer_size; ++outer)
    {
        for (int i = 0; i < param->indices_num; i++)
        {
            memcpy(out_ptr + (outer * param->indices_num + i) * inner_size,
                   in_ptr + (outer * axis_size + ( int )input_indices[i]) * inner_size, inner_size * sizeof(float));
        }
    }

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct gather_param* gather_param = ( struct gather_param* )ir_node->op.param_mem;
    gather_param_t* op_priv_info = ( gather_param_t* )exec_node->ops_priv;

    op_priv_info->axis = gather_param->axis;
    op_priv_info->indices_num = gather_param->indices_num;

    /* prerun now */
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct ir_tensor* indices_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    gather_param_t* op_priv_info = ( gather_param_t* )exec_node->ops_priv;

    int out_size = input_tensor->elem_num;

    // auto in_dim = input_tensor->GetShape().GetDim();
    void* input = input_tensor->data;

    void* indices_data = indices_tensor->data;

    op_priv_info->dim_size = input_tensor->dim_num;

    for (int i = 0; i < op_priv_info->dim_size; i++)
    {
        op_priv_info->in_shape[i] = input_tensor->dims[i];
    }

    // int indices_num = op_param.indices_num;
    void* output = output_tensor->data;

    int ret = ref_gather_fp32(input, indices_data, output, op_priv_info, exec_graph->num_thread);

    return ret;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;

    gather_param_t* op_priv_info = ( gather_param_t* )sys_malloc(sizeof(gather_param_t));

    if (op_priv_info == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(op_priv_info, 0, sizeof(gather_param_t));

    exec_node->ops_priv = op_priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    gather_param_t* op_priv_info = ( gather_param_t* )exec_node->ops_priv;

    sys_free(op_priv_info);

    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops gather_node_ops = {.prerun = prerun,
                                          .run = run,
                                          .reshape = NULL,
                                          .postrun = NULL,
                                          .init_node = init_node,
                                          .release_node = release_node,
                                          .score = score};

static int reg_gather_ops(void* arg)
{
    return register_builtin_node_ops(OP_GATHER, &gather_node_ops);
}

static int unreg_gather_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_GATHER, &gather_node_ops);
}

AUTO_REGISTER_OPS(reg_gather_ops);
AUTO_UNREGISTER_OPS(unreg_gather_ops);
