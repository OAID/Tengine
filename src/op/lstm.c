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
#include <stdbool.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "lstm_param.h"

#include "vector.h"

DEFINE_PARM_PARSE_ENTRY(lstm_param, forget_bias, clip, output_len, sequence_len, input_size, hidden_size, cell_size,
                        has_projection, has_peephole, has_clip, has_bias, has_init_state);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct lstm_param* lstm_param = ( struct lstm_param* )(node->op.param_mem);
    int batch_size = input->dims[1];
    if (lstm_param->mxnet_flag == 0)
    {
        batch_size = input->dims[0];
    }
    int dims[4];
    if (lstm_param->mxnet_flag == 0)
    {
        dims[0] = input->dims[0];
        dims[1] = 1;
        dims[2] = input->dims[1];
        dims[3] = lstm_param->hidden_size;
    }
    else
    {
        dims[0] = input->dims[0];
        dims[1] = batch_size;
        dims[2] = lstm_param->hidden_size;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}
static int init_op(struct ir_op* op)
{
    lstm_param_t* lstm_param = ( lstm_param_t* )sys_malloc(sizeof(lstm_param_t));

    if (lstm_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    lstm_param->forget_bias = 0;
    lstm_param->clip = 0;
    lstm_param->output_len = 1;
    lstm_param->sequence_len = 1;
    lstm_param->input_size = 1;
    lstm_param->hidden_size = 1;
    lstm_param->cell_size = 1;
    lstm_param->has_projection = 0;
    lstm_param->has_peephole = 0;
    lstm_param->has_clip = 0;
    lstm_param->has_bias = 0;
    lstm_param->has_init_state = 0;

    op->param_mem = lstm_param;
    op->param_size = sizeof(lstm_param_t);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_lstm_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_LSTM, OP_LSTM_NAME, &m);
}

static int unregister_lstm_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(lstm_param));
    return unregister_op(OP_LSTM, 1);
}

AUTO_REGISTER_OP(register_lstm_op);
AUTO_UNREGISTER_OP(unregister_lstm_op);
