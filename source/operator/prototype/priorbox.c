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

#include "priorbox_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    priorbox_param_t* priorbox_param = ( priorbox_param_t* )node->op.param_mem;

    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    // out shape [feat_width,feat_height,num_priors_ * 4,2]
    int len_aspect_ratio = 1;
    if (priorbox_param->flip)
    {
        len_aspect_ratio += 1;
    }

    int len_max = 0;
    if (priorbox_param->max_size_num > 0)
    {
        if (priorbox_param->max_size_num == priorbox_param->min_size_num)
        {
            len_max += 1;
        }
        else
        {
            // max_size_len must equal min_size_len
            return -1;
        }
    }

    priorbox_param->num_priors = (priorbox_param->aspect_ratio_size * len_aspect_ratio + 1 + len_max) * priorbox_param->min_size_num;

    priorbox_param->out_dim = input->dims[2] * input->dims[3] * priorbox_param->num_priors * 4;

    int dims[4];
    dims[0] = input->dims[0];
    dims[1] = 2;
    dims[2] = priorbox_param->out_dim;
    dims[3] = 1;

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct priorbox_param* priorbox_param = ( struct priorbox_param* )sys_malloc(sizeof(struct priorbox_param));

    if (priorbox_param == NULL)
    {
        return -1;
    }

    priorbox_param->offset = 0.5f;

    op->param_mem = priorbox_param;
    op->param_size = sizeof(struct priorbox_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct priorbox_param* priorbox_param = ( struct priorbox_param* )op->param_mem;

    if (priorbox_param->aspect_ratio)
        sys_free(priorbox_param->aspect_ratio);
    if (priorbox_param->max_size)
        sys_free(priorbox_param->max_size);
    if (priorbox_param->min_size)
        sys_free(priorbox_param->min_size);
    if (priorbox_param->variance)
        sys_free(priorbox_param->variance);

    sys_free(op->param_mem);
}


int register_priorbox_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_PRIORBOX, OP_PRIORBOX_NAME, &m);
}


int unregister_priorbox_op()
{
    return unregister_op(OP_PRIORBOX, 1);
}
