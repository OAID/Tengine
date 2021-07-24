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
 * Author: qli@openailab.com
 */

#include "rnn_param.h"

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


static int init_op(struct op* op)
{
    struct rnn_param* rnn_param = ( struct rnn_param* )sys_malloc(sizeof(struct rnn_param));

    if (rnn_param == NULL)
    {
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


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_rnn_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_RNN, OP_RNN_NAME, &m);
}


int unregister_rnn_op()
{
    return unregister_op(OP_RNN, 1);
}
