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
 * Author: bhu@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "psroipooling_param.h"

DEFINE_PARM_PARSE_ENTRY(psroipooling_param, pooled_w, pooled_h, spatial_scale, output_dim);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct psroipooling_param* psroipooling_param = ( struct psroipooling_param* )(node->op.param_mem);

    int output_n = input->dims[0];
    int output_c = psroipooling_param->output_dim;
    int output_h = psroipooling_param->pooled_h;
    int output_w = psroipooling_param->pooled_w;

    int out_dim[4];

    out_dim[0] = output_n;
    out_dim[1] = output_c;
    out_dim[2] = output_h;
    out_dim[3] = output_w;

    set_ir_tensor_shape(output, out_dim, 4);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct psroipooling_param* psroipooling_param =
        ( struct psroipooling_param* )sys_malloc(sizeof(struct psroipooling_param));

    if (psroipooling_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    psroipooling_param->pooled_w = 0;
    psroipooling_param->pooled_h = 0;
    psroipooling_param->spatial_scale = 0.f;
    psroipooling_param->output_dim = 0;

    op->param_mem = psroipooling_param;
    op->param_size = sizeof(struct psroipooling_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_psroipooling_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_PSROIPOOLING, OP_PSROIPOOLING_NAME, &m);
}

static int unregister_psroipooling_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(psroipooling_param));
    return unregister_op(OP_PSROIPOOLING, 1);
}

AUTO_REGISTER_OP(register_psroipooling_op);
AUTO_UNREGISTER_OP(unregister_psroipooling_op);
