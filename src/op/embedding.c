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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: zpluo@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "embedding_param.h"

DEFINE_PARM_PARSE_ENTRY(embedding_param, num_output, input_dim, bias_term, weight_data_size);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dims[2] = {0};
    dims[0] = 1;
    for (int ii = 0; ii < input->dim_num; ++ii)
    {
        dims[0] *= input->dims[ii];
    }

    struct embedding_param* param = ( struct embedding_param* )node->op.param_mem;

    dims[1] = param->num_output;

    set_ir_tensor_shape(output, dims, 2);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct embedding_param* param = ( struct embedding_param* )sys_malloc(sizeof(struct embedding_param));

    if (param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    memset(param, 0, sizeof(struct embedding_param));
    op->param_mem = param;
    op->param_size = sizeof(struct embedding_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_embedding_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_EMBEDDING, OP_EMBEDDING_NAME, &m);
}

static int unregister_embedding_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(embedding_param));
    return unregister_op(OP_EMBEDDING, 1);
}

AUTO_REGISTER_OP(register_embedding_op);
AUTO_UNREGISTER_OP(unregister_embedding_op);
