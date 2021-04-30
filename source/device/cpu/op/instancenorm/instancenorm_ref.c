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
 * Author: zpluo@openailab.com
 */

#include "instancenorm_param.h"

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

int ref_instancenorm_fp32(float* input_data, float* output_data, float* gamma_data, float* beta_data, int size,
                          int channels, int n, float eps, float scale, float zero_point, int layout)
{
    int image_size = channels * size;
    float sum = 0.f;
    float sqsum = 0.f;
    int offset = 0;
    for (int s = 0; s < n; s++)
    {
        for (int i = 0; i < channels; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                sum += input_data[offset];
            }
            float mean = sum / size;
            float tmp = 0.f;
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                tmp = input_data[offset] - mean;
                sqsum += tmp * tmp;
            }
            float var = sqsum / size;

            float a = gamma_data[i] / (sqrt(var + eps));
            float b = -mean * a + beta_data[i];
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                output_data[offset] = input_data[offset] * a + b;
            }
        }
    }
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* gamma_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* beta_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);

    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int size = w * h;

    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;
    void* beta_data = beta_tensor->data;
    void* gamma_data = gamma_tensor->data;

    struct instancenorm_Param* param = ( struct instancenorm_Param* )node->op.param_mem;
    float eps = param->eps;
    float scale = 1.f;
    int zero_point = 0;

    return ref_instancenorm_fp32(in_data, out_data, gamma_data, beta_data, size, c, n, eps, scale, zero_point, 0);
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
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

int register_instancenorm_ref_op()
{
    return register_builtin_node_ops(OP_INSTANCENORM, &hcl_node_ops);
}

int unregister_instancenorm_ref_op()
{
    return unregister_builtin_node_ops(OP_INSTANCENORM, &hcl_node_ops);
}
