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

#include "clip_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

#include "float.h"

static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    if (node->input_num == 3)
    {
        struct tensor* clip_min = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);
        struct tensor* clip_max = get_ir_graph_tensor(ir_graph, node->input_tensors[2]);

        struct clip_param* clip_param = (struct clip_param*)node->op.param_mem;
        float* min = (float*)clip_min->data;
        float* max = (float*)clip_max->data;
        if (min && clip_min->elem_num > 0)
        {
            clip_param->min = min[0];
        }
        if (max && clip_max->elem_num > 0)
        {
            clip_param->max = max[0];
        }
    }

    set_ir_tensor_shape(output, input->dims, input->dim_num);

    return 0;
}

static int init_op(struct op* op)
{
    struct clip_param* clip_param = (struct clip_param*)sys_malloc(sizeof(struct clip_param));

    if (clip_param == NULL)
    {
        return -1;
    }

    /* set the param default value */
    clip_param->max = FLT_MAX;
    clip_param->min = -FLT_MAX;

    op->param_mem = clip_param;
    op->param_size = sizeof(struct clip_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}

int register_clip_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_CLIP, OP_CLIP_NAME, &m);
}

int unregister_clip_op()
{
    return unregister_op(OP_CLIP, 1);
}
