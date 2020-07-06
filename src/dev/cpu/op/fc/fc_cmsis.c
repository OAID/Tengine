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
 * Author: haitao@openailab.com
 */

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "cpu_node_ops.h"
#include "tengine_op.h"
#include "op/pool_param.h"

struct cmsis_param
{
    uint16_t bias_shift;
    uint16_t out_shift;
};

void arm_maxpool_q7_HWC_nonsquare(q7_t* Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
                                  const uint16_t ch_im_in, const uint16_t dim_kernel, const uint16_t padding,
                                  const uint16_t stride, const uint16_t dim_im_out_x, const uint16_t dim_im_out_y,
                                  q7_t* bufferA, q7_t* Im_out);

static inline int cal_shift(int scale)
{
    int shift = 0;

    while ((1 << shift) < scale)
        shift++;

    return shift;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    int bias_shift = 0;
    int out_shift = 0;

    if (ir_node->input_num > 2)
    {
        struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int scale = ir_tensor->scale;
        bias_shift = cal_shift(scale);
    }

    struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int scale = ir_tensor->scale;
    out_shift = cal_shift(scale);

    struct cmsis_param* param = ( struct cmsis_param* )sys_malloc(sizeof(struct cmsis_param));

    param->bias_shift = bias_shift;
    param->out_shift = out_shift;

    exec_node->ops_priv = param;

    struct ir_tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    exec_node->shared_mem_size = weight_tensor->dims[1] * 2;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* weight_tensor;
    struct ir_tensor* bias_tensor = NULL;
    struct ir_tensor* output_tensor;
    struct cmsis_param* cmsis_param = ( struct cmsis_param* )exec_node->ops_priv;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    int ret =
        arm_fully_connected_q7(input_tensor->data, weight_tensor->data, weight_tensor->dims[1], weight_tensor->dims[0],
                               cmsis_param->bias_shift, cmsis_param->out_shift, bias_tensor ? bias_tensor->data : NULL,
                               output_tensor->data, exec_graph->shared_mem);

    if (ret != ARM_MATH_SUCCESS)
        return -1;

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return -1;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops cmsis_node_ops = {.prerun = NULL,
                                         .run = run,
                                         .reshape = reshape,
                                         .postrun = NULL,
                                         .init_node = init_node,
                                         .release_node = release_node,
                                         .score = score};

static int reg_fc_cmsis_ops(void* arg)
{
    return register_builtin_node_ops(OP_FC, &cmsis_node_ops);
}

static int unreg_fc_cmsis_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_FC, &cmsis_node_ops);
}

AUTO_REGISTER_OPS(reg_fc_cmsis_ops);
AUTO_UNREGISTER_OPS(unreg_fc_cmsis_ops);
