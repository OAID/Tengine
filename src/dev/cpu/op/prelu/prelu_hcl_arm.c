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
#include "cortex_a/prelu_kernel_arm.h"

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

    int ret = -1;
    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    void* data = input_tensor->data;
    void* out_data = output_tensor->data;
    void* slope = slope_tensor->data;

    ret = prelu_kernel_run(data, out_data, dim0, dim1, dim2, dim3, slope, layout, exec_graph->num_thread);

    if (ret < 0)
        return -1;
    else
        return 0;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = NULL,
                                       .release_node = NULL,
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
