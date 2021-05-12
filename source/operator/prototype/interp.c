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

#include "interp_param.h"

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
    int in_n = input->dims[0];
    int in_c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    struct interp_param* param = ( struct interp_param* )(node->op.param_mem);

    if (param == NULL)
    {
        return -1;
    }

    if (param->height_scale != 0 && param->width_scale != 0)
    {
        param->output_height = in_h * param->height_scale;
        param->output_width = in_w * param->width_scale;
    }
    else
    {
        param->height_scale = (float )param->output_height / (float )in_h;
        param->width_scale = (float )param->output_width / (float )in_w;
    }

    int dim[4] = {0};

    dim[0] = in_n;
    dim[1] = in_c;
    dim[2] = param->output_height;
    dim[3] = param->output_width;

    set_ir_tensor_shape(output, dim, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct interp_param* interp_param = ( struct interp_param* )sys_malloc(sizeof(struct interp_param));

    if (interp_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    interp_param->resize_type = 1;
    interp_param->output_height = 0;
    interp_param->output_width = 0;
    interp_param->height_scale = 1.f;
    interp_param->width_scale = 1.f;

    op->param_mem = interp_param;
    op->param_size = sizeof(struct interp_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_interp_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_INTERP, OP_INTERP_NAME, &m);
}


int unregister_interp_op()
{
    return unregister_op(OP_INTERP, 1);
}
