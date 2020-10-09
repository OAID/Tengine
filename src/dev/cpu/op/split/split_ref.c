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

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "split_param.h"

int ref_split_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct split_param* split_param, int* slice_index, int num_slices, int slice_size, int in_slice, int slice_axis)
{
    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    if (split_param->is_caffe)
    {
        memcpy(output_data, input_data, input_tensor->elem_num * sizeof(float));
    }
    else
    {
        int out_slice = 0;

        out_slice = output_tensor->dims[slice_axis];

        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + *slice_index) * slice_size;
            int out_offset = n * out_slice * slice_size;
            memcpy(output_data + out_offset, input_data + in_offset, slice_size * out_slice * sizeof(float));
        }

        *slice_index += out_slice;
    }
	
	return 0;
}

int ref_split_uint8(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct split_param* split_param, int* slice_index, int num_slices, int slice_size, int in_slice, int slice_axis)
{
    uint8_t* input_data = input_tensor->data;
    uint8_t* output_data = output_tensor->data;

    if (split_param->is_caffe)
    {
        memcpy(output_data, input_data, input_tensor->elem_num * sizeof(uint8_t));
    }
    else
    {
        int out_slice = 0;

        out_slice = output_tensor->dims[slice_axis];

        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + *slice_index) * slice_size;
            int out_offset = n * out_slice * slice_size;
            memcpy(output_data + out_offset, input_data + in_offset, slice_size * out_slice * sizeof(uint8_t));
        }

        *slice_index += out_slice;
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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    // struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    // output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct split_param* split_param = ( struct split_param* )ir_node->op.param_mem;

    /* the follow codes need to be checked ! */
    int slice_axis = split_param->axis;
    int num_slices = 1;
    int slice_size = 1;

    for (int i = 0; i < slice_axis; i++)
        num_slices = num_slices * input_tensor->dims[i];

    for (int i = slice_axis + 1; i < input_tensor->dim_num; i++)
        slice_size = slice_size * input_tensor->dims[i];

    int in_slice = input_tensor->dims[slice_axis];
    int slice_index = 0;
    int out_num = ir_node->output_num;

	int ret = -1;
    for (int i = 0; i < out_num; i++)
    {
        struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
        
        if (input_tensor->data_type == TENGINE_DT_FP32)
            ret = ref_split_fp32(input_tensor, output_tensor, split_param, &slice_index, num_slices, slice_size, in_slice, slice_axis);
       	else if(input_tensor->data_type == TENGINE_DT_UINT8)
           	ret = ref_split_uint8(input_tensor, output_tensor, split_param, &slice_index, num_slices, slice_size, in_slice, slice_axis);
    }

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_split_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_SPLIT, &hcl_node_ops);
}

static int unreg_split_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_SPLIT, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_split_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_split_hcl_ops);
