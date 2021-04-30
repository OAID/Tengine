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
 * Author: xlchen@openailab.com
 */

#include "argmin_param.h"

#include "defines.h"
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

    struct argmin_param* argmin_param = ( struct argmin_param* )(node->op.param_mem);

    int axis = argmin_param->axis;

    if (axis >= input->dim_num)
    {
        return -1;
    }

    int outdims[TE_MAX_SHAPE_DIM_NUM * 2];

    // Change HWC to CHW
    int tmp = input->dims[2];
    input->dims[2] = input->dims[1];
    input->dims[1] = input->dims[0];
    input->dims[0] = tmp;
    input->dims[3] = 1;

    if (input->dims[0] != 1)    // input 3 keepdimss
    {
        for (int i = 0, j = 0; i < 3; i++)
        {
            if (i != axis)
                outdims[j++] = input->dims[i];
        }
    }
    else    // input 2 keepdimss
    {
        for (int i = 0, j = 0; i < 4; i++)
            outdims[j++] = input->dims[i];
        outdims[axis + 1] = outdims[axis + 2];
    }
    outdims[2] = outdims[3] = 1;

    if (argmin_param->keepdims == 2)
    {
        // Change CHW to HWC
        tmp = input->dims[0];
        input->dims[0] = input->dims[1];
        input->dims[1] = input->dims[2];
        input->dims[2] = tmp;
    }

    set_ir_tensor_shape(output, outdims, input->dim_num);

    return 0;
}


static int init_op(struct op* op)
{
    struct argmin_param* argmin_param = ( struct argmin_param* )sys_malloc(sizeof(struct argmin_param));

    if (argmin_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    argmin_param->axis = 0;
    argmin_param->keepdims = 1;

    op->param_mem = argmin_param;
    op->param_size = sizeof(struct argmin_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_argmin_op()
{
    struct method m;
    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_ARGMIN, OP_ARGMIN_NAME, &m);
}


int unregister_argmin_op()
{
    return unregister_op(OP_ARGMIN, 1);
}
