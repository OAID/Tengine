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
 * Author: bzhang@openailab.com
 */

#include "spatialtransformer_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"

#include <string.h>

static int infer_shape(struct node* node)
{
    struct spatialtransformer_param* param = (struct spatialtransformer_param*)(node->op.param_mem);

    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct vector* new_shape = create_vector(sizeof(int), NULL);
    int dim_size = 2;
    for (int i = 0; i < dim_size; i++)
    {
        int shape = param->target_shape[i];
        push_vector_data(new_shape, (void*)&shape);
    }

    int out_dim_size = 4;
    int* new_shape_temp = (int*)sys_malloc(out_dim_size * sizeof(int));

    if (dim_size == 2)
    {
        for (int i = 0; i < get_vector_num(new_shape); i++)
        {
            int* a = (int*)get_vector_data(new_shape, i);
            new_shape_temp[i + dim_size] = *a;
        }
        new_shape_temp[0] = 1;
        new_shape_temp[1] = input->dims[1];
    }

    output->layout = input->layout;
    int ret = set_ir_tensor_shape(output, new_shape_temp, out_dim_size);

    sys_free(new_shape_temp);
    release_vector(new_shape);
    return ret;
}

static int init_op(struct op* op)
{
    struct spatialtransformer_param* spatialtransformer_param = (struct spatialtransformer_param*)sys_malloc(sizeof(struct spatialtransformer_param));

    if (spatialtransformer_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    memset(spatialtransformer_param, 0, sizeof(struct spatialtransformer_param));

    spatialtransformer_param->sampler_type = -1;
    spatialtransformer_param->transformer_type = -1;

    op->param_mem = spatialtransformer_param;
    op->param_size = sizeof(struct spatialtransformer_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct op* op)
{
    struct spatialtransformer_param* param = (struct spatialtransformer_param*)op->param_mem;

    if (param->target_shape)
        sys_free(param->target_shape);

    sys_free(op->param_mem);
}

int register_spatialtransformer_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SPATIALTRANSFORMER, OP_SPATIALTRANSFORMER_NAME, &m);
}

int unregister_spatialtransformer_op()
{
    return unregister_op(OP_SPATIALTRANSFORMER, 1);
}
