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

#include "split_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct split_param* split_param = ( struct split_param* )(node->op.param_mem);

    int axis = split_param->axis;

    int input_dim[4];
    for (int i = 0; i < input->dim_num; i++)
        input_dim[i] = input->dims[i];

    if (split_param->is_caffe)
    {
        for (int i = 0; i < node->output_num; i++)
        {
            ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
            set_ir_tensor_shape(output, input->dims, input->dim_num);
        }
    }
    else
    {
        if (get_vector_num(split_param->split_sizes_) != 0)
        {
            int sum_check = 0;
            int input_slice_num = input_dim[axis];

            for (int i = 0; i < get_vector_num(split_param->split_sizes_); i++)
            {
                sum_check += (( int* )get_vector_data(split_param->split_sizes_, i))[0];
            }

            if (sum_check != input_slice_num)
            {
                TLOG_ERR("Tengine Fatal: Infer shape for Split failed(%d vs. %d).\n", sum_check, input_slice_num);
                return -1;
            }

            for (int i = 0; i < get_vector_num(split_param->split_sizes_); i++)
            {
                input_dim[axis] = (( int* )get_vector_data(split_param->split_sizes_, i))[0];
                ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                set_ir_tensor_shape(output, input_dim, input->dim_num);
            }
        }
        else
        {
            int split_dim = split_param->split_dim;
            int split_shape = 0;

            if (input_dim[axis] % split_dim != 0)
            {
                TLOG_ERR("Tengine Fatal: Infer shape for Split failed, input dim can not be divided by split dim with no remainder(%d %% %d).\n", input_dim[axis], split_dim);
                return -1;
            }

            split_shape = input_dim[axis] / split_dim;
            input_dim[axis] = split_shape;

            if (split_shape == 1)
            {
                int output_dim[4];
                for (int i = 0; i < input->dim_num - 1; i++)
                {
                    if (i >= axis)
                        output_dim[i] = input_dim[i + axis];
                    else
                        output_dim[i] = input_dim[i];
                }

                for (int i = 0; i < node->output_num; i++)
                {
                    ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                    set_ir_tensor_shape(output, input->dims, input->dim_num - 1);
                }
            }

            for (int i = 0; i < node->output_num; i++)
            {
                ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[i]);
                set_ir_tensor_shape(output, input->dims, input->dim_num);
            }
        }
    }

    return 0;
}


static int init_op(ir_op_t* op)
{
    struct split_param* split_param = ( struct split_param* )sys_malloc(sizeof(struct split_param));

    if (split_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    split_param->axis = 0;
    split_param->split_dim = 1;
    split_param->is_caffe = 0;
    split_param->is_onnx = 0;
    split_param->split_sizes_ = NULL;

    op->param_mem = split_param;
    op->param_size = sizeof(struct split_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    struct split_param* split_param = ( struct split_param* )op->param_mem;

    if (split_param->split_sizes_)
        release_vector(split_param->split_sizes_);

    sys_free(op->param_mem);
}


int register_split_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SPLIT, OP_SPLIT_NAME, &m);
}


int unregister_split_op()
{
    return unregister_op(OP_SPLIT, 1);
}
