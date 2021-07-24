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
 * Author: chh@openailab.com
 */

#include "crop_param.h"

#include "defines.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[1]); // Don't try to modify !
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct crop_param* crop_param = ( struct crop_param* )(node->op.param_mem);

    int input_h = input->dims[2];
    int input_w = input->dims[3];
    int output_h = 0;
    int output_w = 0;

    // MXNet
    if (crop_param->flag == 1)
    {
        if (crop_param->num_args == 2)
        {
            output_h = input_h;
            output_w = input_w;
        }
        if (crop_param->num_args == 1)
        {
            output_h = crop_param->crop_h;
            output_w = crop_param->crop_w;
        }
    }
    // Caffe
    if (crop_param->flag == 0)
    {
        output_h = input_h;
        output_w = input_w;
    }

    int out_size = input->dim_num;
    int out_dim[4];

    out_dim[0] = input->dims[0];
    out_dim[1] = input->dims[1];
    out_dim[2] = output_h;
    out_dim[3] = output_w;

    set_ir_tensor_shape(output, out_dim, out_size);

    return 0;
}


static int init_op(struct op* op)
{
    struct crop_param* crop_param = ( struct crop_param* )sys_malloc(sizeof(struct crop_param));

    if (crop_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    crop_param->num_args = 0;
    crop_param->offset_c = 0;
    crop_param->offset_h = 0;
    crop_param->offset_w = 0;
    crop_param->crop_h = 0;
    crop_param->crop_w = 0;
    crop_param->center_crop = 0;
    crop_param->axis = 2;
    crop_param->flag = 0;

    op->param_mem = crop_param;
    op->param_size = sizeof(struct crop_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_crop_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_CROP, OP_CROP_NAME, &m);
}


int unregister_crop_op()
{
    return unregister_op(OP_CROP, 1);
}
