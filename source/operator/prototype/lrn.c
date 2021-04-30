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

#include "lrn_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    set_ir_tensor_shape(output, input->dims, input->dim_num);

    return 0;
}


static int init_op(struct op* op)
{
    struct lrn_param* lrn_param = ( struct lrn_param* )sys_malloc(sizeof(struct lrn_param));

    if (lrn_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    lrn_param->alpha = 1.f;
    lrn_param->beta = 0.75f;
    lrn_param->norm_region = 0;
    lrn_param->k = 1.f;
    lrn_param->local_size = 5;

    op->param_mem = lrn_param;
    op->param_size = sizeof(struct lrn_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_lrn_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_LRN, OP_LRN_NAME, &m);
}


int unregister_lrn_op()
{
    return unregister_op(OP_LRN, 1);
}
