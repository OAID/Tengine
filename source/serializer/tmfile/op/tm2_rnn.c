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

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int rnn_op_map(int op)
{
    return OP_RNN;
}


static int tm2_load_rnn(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct rnn_param* rnn_param = (struct rnn_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv* )ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_RnnParam* tm_param = (TM2_RnnParam* )(mem_base + tm_op->offset_t_param);

    rnn_param->clip = tm_param->clip;
    rnn_param->output_len = tm_param->output_len;
    rnn_param->sequence_len = tm_param->sequence_len;
    rnn_param->input_size = tm_param->input_size;
    rnn_param->hidden_size = tm_param->hidden_size;
    rnn_param->has_clip = tm_param->has_clip;
    rnn_param->has_bias = tm_param->has_bias;
    rnn_param->has_init_state = tm_param->has_init_state;
    rnn_param->activation = tm_param->activation;

    return 0;
}


int register_tm2_rnn_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if(tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_RNN, 1, tm2_load_rnn, rnn_op_map, NULL);

    return 0;
}


int unregister_tm2_rnn_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_RNN, 1, tm2_load_rnn);

    return 0;
}
