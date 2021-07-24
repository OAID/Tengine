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
 * Author: haitao@openailab.com
 */

#include "concat_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* graph = node->graph;
    ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct concat_param* concat_param = ( struct concat_param* )(node->op.param_mem);

    int concat_shape = 0;
    int axis = concat_param->axis;

    /* transpose axis from nhwc to nchw */
    if (graph->model_layout == TENGINE_LAYOUT_NHWC)
    {
        ir_tensor_t* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        if (input_tensor->dim_num == 4)
        {
            if (axis == 1)
                axis = 2;
            else if (axis == 2)
                axis = 3;
            else if (axis == 3)
                axis = 1;
            else
            {
                TLOG_ERR("Tengine Fatal: Infer shape for Concat failed(axis = %d; dim = 3).\n", axis);
                return -1;
            }
            concat_param->axis = axis;
        }
        else if (input_tensor->dim_num == 3)
        {
            if (axis == 1)
                axis = 2;
            else if (axis == 2)
                axis = 1;
            else
            {
                TLOG_ERR("Tengine Fatal: Infer shape for Concat failed(axis = %d; dim = 4).\n", axis);
                return -1;
            }
            concat_param->axis = axis;
        }
    }

    for (int i = 0; i < node->input_num; i++)
    {
        ir_tensor_t* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);

        if (axis < 0)
        {
            axis = input_tensor->dim_num + axis;
            concat_param->axis = axis;
        }

        ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[i]);
        concat_shape += input->dims[axis];
    }

    ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    int dims[4];
    for (int i = 0; i < input->dim_num; i++)
    {
        dims[i] = input->dims[i];
    }

    dims[axis] = concat_shape;
    output->layout = input->layout;
    set_ir_tensor_shape(output, dims, input->dim_num);

    return 0;
}


static int init_op(ir_op_t* op)
{
    struct concat_param* concat_param = ( struct concat_param* )sys_malloc(sizeof(struct concat_param));

    if (concat_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    concat_param->axis = 0;

    op->param_mem = concat_param;
    op->param_size = sizeof(struct concat_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_concat_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_CONCAT, OP_CONCAT_NAME, &m);
}


int unregister_concat_op()
{
    return unregister_op(OP_CONCAT, 1);
}
