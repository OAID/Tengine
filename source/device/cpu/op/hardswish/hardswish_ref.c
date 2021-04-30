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


int ref_hardswish_fp32(struct tensor* input_tensor, struct tensor* output_tensor)
{
    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;
    int size = input_tensor->elem_num;

    for (int i = 0; i < size; i++)
    {
        float tmp = input_data[i] + 3.f;

        if (tmp < 0.f)
            tmp = 0.f;
        if (tmp > 6.f)
            tmp = 6.f;
        
        output_data[i] = input_data[i] * (tmp / 6.f);
    }

    return 0;
}

int ref_hardswish_uint8(struct tensor* input_tensor, struct tensor* output_tensor)
{
    int size = input_tensor->elem_num;

    // dequant
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = sys_malloc(size * sizeof(float));

    for(int i = 0; i < size; i++)
        data_fp32[i] = ((float) input_uint8[i] - (float)input_zero) * input_scale;

    for (int i = 0; i < size; i++)
    {
        float tmp = data_fp32[i] + 3.f;

        if (tmp < 0.f)
            tmp = 0.f;
        if (tmp > 6.f)
            tmp = 6.f;

        data_fp32[i] = data_fp32[i] * (tmp / 6.f);
    }

    // quant
    for(int i=0; i<size; i++)
    {
        int udata = round(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }
    sys_free(data_fp32);

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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_hardswish_fp32(input_tensor, output_tensor);
    else if(input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_hardswish_uint8(input_tensor, output_tensor);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

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
int register_hardswish_ref_op()
{
    return register_builtin_node_ops(OP_HARDSWISH, &hcl_node_ops);
}

int unregister_hardswish_ref_op()
{
    return unregister_builtin_node_ops(OP_HARDSWISH, &hcl_node_ops);
}
