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
 * Author: qli@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "rnn_param.h"

DEFINE_PARM_PARSE_ENTRY(rnn_param, clip, output_len, sequence_len, input_size, hidden_size, has_clip, has_bias,
                        has_init_state, activation);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct rnn_param* rnn_param = ( struct rnn_param* )(node->op.param_mem);
    int dims[3];

    // input tensors:
    // 0 --- input: [seq_length, batch_size,input_size]
    // 1 --- kernel [ (input_size+hidden_size),hidden_state_size]
    // others: optional

    // output tensor: [output_len,batch_size,hidden_size]
    int batch_size = input->dims[1];

    dims[0] = rnn_param->output_len;
    dims[1] = batch_size;
    dims[2] = rnn_param->hidden_size;

    set_ir_tensor_shape(output, dims, 3);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct rnn_param* rnn_param = ( struct rnn_param* )sys_malloc(sizeof(struct rnn_param));

    if (rnn_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }
    rnn_param->inithiddenname = "init_h";
    rnn_param->inithiddenname = "bias";

    op->param_mem = rnn_param;
    op->param_size = sizeof(struct rnn_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_rnn_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_RNN, OP_RNN_NAME, &m);
}

static int unregister_rnn_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(rnn_param));
    return unregister_op(OP_RNN, 1);
}

AUTO_REGISTER_OP(register_rnn_op);
AUTO_UNREGISTER_OP(unregister_rnn_op);
