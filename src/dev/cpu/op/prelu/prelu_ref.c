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
 * Author: hhchen@openailab.com
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

static int ref_prelu_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct ir_tensor* slope_tensor)
{
    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    float* data = input_tensor->data;
    float* out_data = output_tensor->data;
    float* slope = slope_tensor->data;

    int offset = 0;
    // nchw
    for (int i = 0; i < dim0; i++)
    {
        for (int c = 0; c < dim1; c++)
        {
            for (int l = 0; l < dim2; l++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    out_data[offset] = MAX(data[offset], 0) + slope[c] * MIN(data[offset], 0.f);
                }
            }
        }
    }
    return 0;
}

static int ref_prelu_uint8(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct ir_tensor* slope_tensor)
{
    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    uint8_t* data = input_tensor->data;
    uint8_t* out_data = output_tensor->data;
    uint8_t* slope = slope_tensor->data;

    /* dequant */
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float slope_scale = slope_tensor->scale;
    uint32_t input_zero = input_tensor->zero_point;
    uint32_t output_zero = output_tensor->zero_point;
    uint32_t slope_zero = slope_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;
    int slope_size = slope_tensor->elem_num;

    float* input_fp32 = ( float* )sys_malloc(input_size * sizeof(float));
    float* output_fp32 = ( float* )sys_malloc(output_size * sizeof(float));
    float* slope_fp32 = ( float* )sys_malloc(slope_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = (( float )data[i] - ( float )input_zero) * input_scale;
    }
    for (int i = 0; i < slope_size; i++)
    {
        slope_fp32[i] = (( float )slope[i] - ( float )slope_zero) * slope_scale;
    }

    int offset = 0;
    // nchw
    for (int i = 0; i < dim0; i++)
    {
        for (int c = 0; c < dim1; c++)
        {
            for (int l = 0; l < dim2; l++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    // nchw
                    offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[c] * MIN(input_fp32[offset], 0.f);
                }
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(output_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        out_data[i] = udata;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);
    sys_free(slope_fp32);

    return 0;
}

static int ref_prelu_int8(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct ir_tensor* slope_tensor)
{
    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    int8_t* data = input_tensor->data;
    int8_t* out_data = output_tensor->data;
    int8_t* slope = slope_tensor->data;

    /* dequant */
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float slope_scale = slope_tensor->scale;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;
    int slope_size = slope_tensor->elem_num;

    float* input_fp32 = ( float* )sys_malloc(input_size * sizeof(float));
    float* output_fp32 = ( float* )sys_malloc(output_size * sizeof(float));
    float* slope_fp32 = ( float* )sys_malloc(slope_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ( float )data[i] * input_scale;
    }
    for (int i = 0; i < slope_size; i++)
    {
        slope_fp32[i] = ( float )slope[i] * slope_scale;
    }

    int offset = 0;
    // nchw
    for (int i = 0; i < dim0; i++)
    {
        for (int c = 0; c < dim1; c++)
        {
            for (int l = 0; l < dim2; l++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    // nchw
                    offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[c] * MIN(input_fp32[offset], 0.f);
                }
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int data_i32 = round(output_fp32[i] / output_scale);
        if (data_i32 > 127)
            data_i32 = 127;
        else if (data_i32 < -127)
            data_i32 = -127;
        out_data[i] = (int8_t)data_i32;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);
    sys_free(slope_fp32);

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

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    slope_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_prelu_fp32(input_tensor, output_tensor, slope_tensor);
    else if(input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_prelu_uint8(input_tensor, output_tensor, slope_tensor);
    else if(input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_prelu_int8(input_tensor, output_tensor, slope_tensor);
    else
        printf("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
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
