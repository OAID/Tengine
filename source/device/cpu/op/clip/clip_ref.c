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

#include "clip_param.h"

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


int ref_clip_fp32(struct tensor* input_tensor, struct tensor* output_tensor, float max, float min, int num_thread)
{
    int total_size = input_tensor->elem_num;
    float* input_data = input_tensor->data;
    float* out_data = output_tensor->data;

    for (int i = 0; i < total_size; i++)
    {
        out_data[i] = input_data[i];

        if (out_data[i] > max)
            out_data[i] = max;
        if (out_data[i] < min)
            out_data[i] = min;
    }

    return 0;
}

int ref_clip_uint8(struct tensor* input_tensor, struct tensor* output_tensor, float max, float min, int num_thread)
{
    int total_size = input_tensor->elem_num;
    uint8_t* input_uint8 = ( uint8_t* )input_tensor->data;
    uint8_t* output_uint8 = ( uint8_t* )output_tensor->data;

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int input_zero = input_tensor->zero_point;
    int output_zero = output_tensor->zero_point;

    /* input dequant */
    float* input_fp32 = ( float* )sys_malloc(total_size * sizeof(float));
    float* output_fp32 = ( float* )sys_malloc(total_size * sizeof(float));

    for (uint32_t i = 0; i < input_tensor->elem_num; i++)
        input_fp32[i] = ((float )input_uint8[i] - (float )input_zero) * input_scale;

    for (int i = 0; i < total_size; i++)
    {
        output_fp32[i] = input_fp32[i];

        if (output_fp32[i] > max)
            output_fp32[i] = max;
        if (output_fp32[i] < min)
            output_fp32[i] = min;
    }

    /* output quant */
    for (int i = 0; i < total_size; i++)
    {
        int output_data = (int)roundf(output_fp32[i] / output_scale) + output_zero;
        output_uint8[i] = output_data > 255 ? 255 : output_data;
    }

    sys_free(input_fp32);
    sys_free(output_fp32); 

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct clip_param* clip_param = ( struct clip_param* )ir_node->op.param_mem;

    int in_size = input_tensor->elem_num;
    float max = clip_param->max;
    float min = clip_param->min;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_clip_fp32(input_tensor, output_tensor, max, min, exec_graph->num_thread);
    else
        ret = ref_clip_uint8(input_tensor, output_tensor, max, min, exec_graph->num_thread);

    return ret;
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

int register_clip_ref_op()
{
    return register_builtin_node_ops(OP_CLIP, &hcl_node_ops);
}

int unregister_clip_ref_op()
{
    return unregister_builtin_node_ops(OP_CLIP, &hcl_node_ops);
}
