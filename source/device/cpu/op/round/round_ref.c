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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

int ref_round_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    // dims size = 2 or 3
    if (input_tensor->dim_num < 4)
    {
        float* input_data = (float*)input_tensor->data;
        float* out_data = (float*)output_tensor->data;
        int total_size = input_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            input_data[i] = round(out_data[i]);
        }

        return 0;
    }
    // dims size 3
    else if (input_tensor->dim_num == 4)
    {
        int w = input_tensor->dims[3];
        int h = output_tensor->dims[2];
        int channels = input_tensor->dims[1];
        int size = h * w;
        int c_step = h * w;

        float* input_data = (float*)input_tensor->data;
        float* out_data = (float*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src = input_data + c_step * q;
            float* dst = out_data + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = round(src[i]);
            }
        }

        return 0;
    }

    return -1;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map[0] = 0;
    // exec_node->inplace_map[1] = 0;
    // exec_node->inplace_map_num = 1;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // exec_node->inplace_map_num = 0;
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    // inplace inference
    // if(input_tensor->data != output_tensor->data)
    // {
    //     TLOG_ERR("input and output are not the same mem\n");
    //     set_tengine_errno(EFAULT);
    //     return -1;
    // }

    int ret = ref_round_fp32(input_tensor, output_tensor, exec_graph->num_thread);
    if (ret != 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
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

int register_round_ref_op()
{
    return register_builtin_node_ops(OP_ROUND, &hcl_node_ops);
}

int unregister_round_ref_op()
{
    return unregister_builtin_node_ops(OP_ROUND, &hcl_node_ops);
}
