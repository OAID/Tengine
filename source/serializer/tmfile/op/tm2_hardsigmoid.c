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

#include "hardsigmoid_param.h"

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
    return OP_HARDSIGMOID;
}


static int tm2_load_hard_sigmoid(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                                 const TM2_Operator* tm_op)
{
    struct hard_sigmoid_param* gather_param = ( struct hard_sigmoid_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_HardsigmoidParam* tm_param = ( TM2_HardsigmoidParam* )(mem_base + tm_op->offset_t_param);

    gather_param->alpha = tm_param->alpha;
    gather_param->beta = tm_param->beta;

    return 0;
}


int register_tm2_hardsigmoid_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_HARDSIGMOID, 1, tm2_load_hard_sigmoid, gather_op_map, NULL);

    return 0;
}


int unregister_tm2_hardsigmoid_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_HARDSIGMOID, 1, tm2_load_hard_sigmoid);

    return 0;
}
