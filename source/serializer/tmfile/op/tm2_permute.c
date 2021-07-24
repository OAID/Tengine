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

#include "permute_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"

static int permute_op_map(int op)
{
    return OP_PERMUTE;
}

static int tm2_load_permute(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct permute_param* permute_param = (struct permute_param*)ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_PermuteParam* tm_param = (TM2_PermuteParam*)(mem_base + tm_op->offset_t_param);

    permute_param->flag = tm_param->flag;
    permute_param->order0 = tm_param->order0;
    permute_param->order1 = tm_param->order1;
    permute_param->order2 = tm_param->order2;
    permute_param->order3 = tm_param->order3;

    return 0;
}

int register_tm2_permute_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_PERMUTE, 1, tm2_load_permute, permute_op_map, NULL);

    return 0;
}

int unregister_tm2_permute_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_PERMUTE, 1, tm2_load_permute);

    return 0;
}
