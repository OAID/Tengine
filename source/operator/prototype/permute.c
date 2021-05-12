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

#include "permute_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    permute_param_t* param = ( struct permute_param* )(node->op.param_mem);

    int dims[TE_MAX_SHAPE_DIM_NUM] = {0};
    int dim_size = input->dim_num;

    if ((param->order0 == 0) && (param->order1 == 2) && (param->order2 == 3) && (param->order3 == 1))
    {
        dims[0] = input->dims[0];
        dims[1] = input->dims[2];
        dims[2] = input->dims[3];
        dims[3] = input->dims[1];

        output->layout = TENGINE_LAYOUT_NHWC;
    }
    else if ((param->order0 == 1) && (param->order1 == 0) && (param->order2 == 2) && dim_size == 3)
    {
        dims[0] = input->dims[1];
        dims[1] = input->dims[0];
        dims[2] = input->dims[2];
    }
    else
    {
        return -1;
    }

    set_ir_tensor_shape(output, dims, dim_size);

    return 0;
}


static int init_op(struct op* op)
{
    struct permute_param* permute_param = ( struct permute_param* )sys_malloc(sizeof(struct permute_param));

    if (permute_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    permute_param->flag = 0;
    permute_param->order0 = 0;
    permute_param->order1 = 1;
    permute_param->order2 = 2;
    permute_param->order3 = 3;
    op->param_mem = permute_param;
    op->param_size = sizeof(struct permute_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_permute_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_PERMUTE, OP_PERMUTE_NAME, &m);
}


int unregister_permute_op()
{
    return unregister_op(OP_PERMUTE, 1);
}
