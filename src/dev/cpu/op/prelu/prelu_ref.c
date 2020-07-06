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
 * Author: bhu@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int ref_prelu_fp32(float* data, float* out_data, int dim0, int dim1, int dim2, int dim3, float* slope,
                          int layout)
{
    int offset = 0;
    // nchw
    // nhwc
    for (int i = 0; i < dim0; i++)
    {
        for (int c = 0; c < dim1; c++)
        {
            for (int l = 0; l < dim2; l++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    if (layout == 0)
                    {
                        // nchw
                        offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    }
                    else
                    {
                        // nhwc
                        offset = i * dim1 * dim2 * dim3 + l * dim3 * dim1 + k * dim1 + c;
                    }
                    out_data[offset] = MAX(data[offset], 0) + slope[c] * MIN(data[offset], 0.f);
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

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] ||
        input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;
    struct ir_tensor* slope_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    slope_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    void* data = input_tensor->data;
    void* out_data = output_tensor->data;
    void* slope = slope_tensor->data;

    int ret = ref_prelu_fp32(data, out_data, dim0, dim1, dim2, dim3, slope, layout);
    if (0 != ret)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_prelu_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_PRELU, &hcl_node_ops);
}

static int unreg_prelu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_PRELU, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_prelu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_prelu_hcl_ops);
