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

#include "embedding_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int gather_op_map(int op)
{
    return OP_EMBEDDING;
}


static int tm2_load_embedding(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                              const TM2_Operator* tm_op)
{
    struct embedding_param* gather_param = ( struct embedding_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_EmbedParam* tm_param = ( TM2_EmbedParam* )(mem_base + tm_op->offset_t_param);

    // gather_param->bias_term = tm_param->bias_term;
    gather_param->input_dim = tm_param->input_dim;
    gather_param->num_output = tm_param->num_output;
    gather_param->weight_data_size = tm_param->weight_data_size;

    return 0;
}


int register_tm2_embedding_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_EMBED, 1, tm2_load_embedding, gather_op_map, NULL);

    return 0;
}


int unregister_tm2_embedding_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_EMBED, 1, tm2_load_embedding);

    return 0;
}
