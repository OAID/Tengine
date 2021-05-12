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

#include "lstm_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int lstm_op_map(int op)
{
    return OP_LSTM;
}


static int tm2_load_lstm(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                         const TM2_Operator* tm_op)
{
    struct lstm_param* lstm_param = ( struct lstm_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_LstmParam* tm_param = ( TM2_LstmParam* )(mem_base + tm_op->offset_t_param);

    lstm_param->forget_bias = tm_param->forget_bias;
    lstm_param->clip = tm_param->clip;
    lstm_param->output_len = tm_param->output_len;
    lstm_param->sequence_len = tm_param->sequence_len;
    lstm_param->input_size = tm_param->input_size;
    lstm_param->hidden_size = tm_param->hidden_size;
    lstm_param->cell_size = tm_param->cell_size;
    lstm_param->has_peephole = tm_param->has_peephole;
    lstm_param->has_projection = tm_param->has_projection;
    lstm_param->has_clip = tm_param->has_clip;
    lstm_param->has_bias = tm_param->has_bias;
    lstm_param->has_init_state = tm_param->has_init_state;
    lstm_param->forget_act = tm_param->forget_act;
    lstm_param->input_act = tm_param->input_act;
    lstm_param->output_act = tm_param->output_act;
    lstm_param->cellin_act = tm_param->cellin_act;
    lstm_param->cellout_act = tm_param->cellout_act;
    lstm_param->mxnet_flag = tm_param->mxnet_flag;

    return 0;
}


int register_tm2_lstm_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_LSTM, 1, tm2_load_lstm, lstm_op_map, NULL);

    return 0;
}


int unregister_tm2_lstm_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_LSTM, 1, tm2_load_lstm);

    return 0;
}
