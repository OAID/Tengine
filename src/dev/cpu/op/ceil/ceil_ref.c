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
 * Update: hhchen@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include <math.h>

int ref_ceil_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, int num_thread)
{
    // dims size = 2 or 3
    if (input_tensor->dim_num < 4)
    {
        float* input_data = input_tensor->data;
        float* out_data = output_tensor->data;
        int total_size = input_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            input_data[i] = ceil(out_data[i]);
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

        float* input_data = input_tensor->data;
        float* out_data = output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src = input_data + c_step * q;
            float* dst = out_data + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = ceil(src[i]);
            }
        }

        return 0;
    }

    return -1;
}

int ref_ceil_uint8(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, int num_thread)
{
     /* dequant */
    uint8_t* input_uint8 = input_tensor->data;
    uint8_t* output_uint8 = output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_data = ( float* )sys_malloc(input_size * sizeof(float));
    float* out_data = ( float* )sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_data[i] = (( float )input_uint8[i] - ( float )input_zero) * input_scale;
    }

    // dims size = 2 or 3
    if (input_tensor->dim_num < 4)
    {
        int total_size = input_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            input_data[i] = ceil(out_data[i]);
        }

        // return 0;
    }
    // dims size 3
    else if (input_tensor->dim_num == 4)
    {
        int w = input_tensor->dims[3];
        int h = output_tensor->dims[2];
        int channels = input_tensor->dims[1];
        int size = h * w;
        int c_step = h * w;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src = input_data + c_step * q;
            float* dst = out_data + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = ceil(src[i]);
            }
        }

        // return 0;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input_data);
    sys_free(out_data);

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
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_ceil_fp32(input_tensor, output_tensor, exec_graph->num_thread);
    else if(input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_ceil_uint8(input_tensor, output_tensor, exec_graph->num_thread);
    
    if (ret != 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
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

static int reg_ceil_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_CEIL, &hcl_node_ops);
}

static int unreg_ceil_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_CEIL, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_ceil_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_ceil_hcl_ops);
