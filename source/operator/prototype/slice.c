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

#include "slice_param.h"

#include "defines.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/sys_port.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* ir_graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct slice_param* slice_param = ( struct slice_param* )(node->op.param_mem);
    int dims_len = input->dim_num;
    int dims_in[TE_MAX_SHAPE_DIM_NUM * 2];

    for (int j = 0; j < dims_len; j++)
    {
        dims_in[j] = input->dims[j];
    }

    if (slice_param->iscaffe)
    {
        int slice_axis = slice_param->axis;

        if (get_vector_num(slice_param->slice_point_) != 0)
        {
            int prev = 0;
            int input_slice_num = input->dims[slice_axis];
            unsigned int i = 0;
            for (; i < slice_param->slice_point_->elem_num; ++i)
            {
                dims_in[slice_axis] = (*( int* )get_vector_data(slice_param->slice_point_, i) - prev);
                prev = *( int* )get_vector_data(slice_param->slice_point_, i);
                set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[i]), dims_in, dims_len);
            }
            // The last one
            dims_in[slice_axis] = (input_slice_num - prev);
            set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[i]), dims_in, dims_len);
        }
        else
        {
            int out_num = node->output_num;
            if (dims_in[slice_axis] % out_num != 0)
                return -1;
            if (slice_axis > ( int )dims_len)
                return -1;
            dims_in[slice_axis] = dims_in[slice_axis] / out_num;
            for (int i = 0; i < out_num; i++)
            {
                set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[i]), dims_in, dims_len);
            }
        }
    }
    else if (slice_param->ismxnet)
    {
        int axis = slice_param->axis;
        int dim_len = input->dim_num;
        // std::vector<int> out_dim(dim_len);
        // out_dim.reserve(input_dim.size());
        int out_dims[TE_MAX_SHAPE_DIM_NUM * 2];
        for (int i = 0; i < dim_len; i++)
        {
            if (i == axis)
            {
                out_dims[i] = slice_param->end - slice_param->begin;
            }
            else
            {
                // int tmpdim=input_dim[i];
                out_dims[i] = dims_in[i];
            }
        }
        set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[0]), out_dims, dim_len);
        // oshape[0].SetDim(out_dim);
        // oshape[0].SetDataLayout(input.GetDataLayout());
    }
    else if (slice_param->isonnx)
    {
        int axis = slice_param->axis;
        int dim_len = input->dim_num;
        int out_dims[TE_MAX_SHAPE_DIM_NUM * 2];
        for (int i = 0; i < dim_len; i++)
        {
            if (i == axis)
            {
                int slice_end = slice_param->end;
                if (slice_param->end > dims_in[i])
                {
                    slice_end = dims_in[i];
                    slice_param->end = slice_end;
                }
                if (slice_end > 0)
                {
                    out_dims[i] = slice_end - slice_param->begin;
                    if (slice_param->step > 1)
                    {
                        out_dims[i] = (out_dims[i] - 1) / slice_param->step + 1;
                    }
                }
                else
                {
                    out_dims[i] = dims_in[i] + (slice_end - slice_param->begin);
                    if (slice_param->step > 1)
                    {
                        out_dims[i] = (out_dims[i] - 1) / slice_param->step + 1;
                    }
                }
                if (0 == out_dims[i])
                    out_dims[i] = dims_in[i];
            }
            else
            {
                out_dims[i] = dims_in[i];
            }
        }
        set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[0]), out_dims, dim_len);
    }
    else
    {
        int dim_len = input->dim_num;
        int out_dims[TE_MAX_SHAPE_DIM_NUM * 2];
        // input shape size must be equal to begin and size's size;
        if ((slice_param->size_->elem_num != slice_param->begin_->elem_num) ||
            (slice_param->size_->elem_num != dim_len))
            return -1;
        for (unsigned int i = 0; i < dim_len; i++)
        {
            out_dims[i] = *( int* )get_vector_data(slice_param->size_, i);
        }
        set_ir_tensor_shape(get_ir_graph_tensor(ir_graph, node->output_tensors[0]), out_dims, dim_len);
    }
    return 0;
}


static int init_op(ir_op_t* op)
{
    slice_param_t* slice_param = ( slice_param_t* )sys_malloc(sizeof(slice_param_t));

    if (slice_param == NULL)
    {
        return -1;
    }

    slice_param->axis = 1;
    slice_param->iscaffe = 0;
    slice_param->ismxnet = 0;
    slice_param->isonnx = 0;

    op->param_mem = slice_param;
    op->param_size = sizeof(struct slice_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    slice_param_t* slice_param = ( slice_param_t* )op->param_mem;

    if (slice_param->slice_point_)
        release_vector(slice_param->slice_point_);
    if (slice_param->begin_)
        release_vector(slice_param->begin_);
    if (slice_param->size_)
        release_vector(slice_param->size_);

    sys_free(op->param_mem);
}


int register_slice_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SLICE, OP_SLICE_NAME, &m);
}


int unregister_slice_op()
{
    return unregister_op(OP_SLICE, 1);
}
