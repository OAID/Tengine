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
 * Author: bhu@openailab.com
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


struct ref_matmul_data
{
    int batch;
    int c;
    int h;
    int w;
    int k;
    int zero[3];    // input, kernel, output
    float scale[3];    // input, kernel, output
};

static int ref_matmul_fp32(const float* input0, float* input1, float* output, struct ref_matmul_data* param)
{
    int batch = param->batch;
    int c = param->c;
    int h = param->h;
    int w = param->w;
    int k = param->k;

    for (int n = 0; n < batch; ++n)
    {
        for (int in_c = 0; in_c < c; in_c++)
        {
            const float* data0 = input0 + n * c * h * w + in_c * h * w;
            float* data1 = input1 + n * c * w * k + in_c * w * k;
            for (int in_h = 0; in_h < h; in_h++)
            {
                for (int in_k = 0; in_k < k; in_k++)
                {
                    float tmp = 0;
                    for (int in_w = 0; in_w < w; in_w++)
                    {
                        int index0 = in_h * w + in_w;
                        int index1 = in_w * k + in_k;
                        tmp += data0[index0] * data1[index1];
                    }
                    *output = tmp;
                    output++;
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
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int dim_size = input_tensor->dim_num;
    struct ref_matmul_data param;
    if (dim_size == 4)
    {
        param.batch = input_tensor->dims[0];
        param.c = input_tensor->dims[1];
        param.h = input_tensor->dims[2];
        param.w = input_tensor->dims[3];
        param.k = input_tensor->dims[3];
    }
    else if (dim_size == 3)
    {
        param.batch = 1;
        param.c = input_tensor->dims[0];
        param.h = input_tensor->dims[1];
        param.w = input_tensor->dims[2];
        param.k = input_tensor->dims[2];
    }
    else if (dim_size == 2)
    {
        param.batch = 1;
        param.c = 1;    // input0->Getse().Shape(0);
        param.h = input_tensor->dims[0];
        param.w = input_tensor->dims[2];
        param.k = input_tensor->dims[2];
    }
    const void* input_data0 = input_tensor->data;
    void* input_data1 = input_tensor1->data;
    void* output_data = output_tensor->data;

    if (ref_matmul_fp32(input_data0, input_data1, output_data, &param) < 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops matmul_node_ops = {.prerun = NULL,
                                          .run = run,
                                          .reshape = NULL,
                                          .postrun = NULL,
                                          .init_node = init_node,
                                          .release_node = release_node,
                                          .score = score};

int register_matmul_ref_op()
{
    return register_builtin_node_ops(OP_MATMUL, &matmul_node_ops);
}

int unregister_matmul_ref_op()
{
    return unregister_builtin_node_ops(OP_MATMUL, &matmul_node_ops);
}
