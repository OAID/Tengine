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
 * Author: bzhang@openailab.com
 */

#include "l2pool_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int l2pool_op_map(int op)
{
    return OP_L2POOL;
}


static int tm2_load_l2pool(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                          const TM2_Operator* tm_op)
{
    struct l2pool_param* l2pool_param = (struct l2pool_param*)ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_L2PoolParam* tm_param = (TM2_L2PoolParam*)(mem_base + tm_op->offset_t_param);
    l2pool_param->paddingType = tm_param->paddingType;
    l2pool_param->kernel_h = tm_param->kernel_h;
    l2pool_param->kernel_w = tm_param->kernel_w;
    l2pool_param->stride_h = tm_param->stride_h;
    l2pool_param->stride_w = tm_param->stride_w;
    return 0;
}


int register_tm2_l2pool_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_L2POOL, 1, tm2_load_l2pool, l2pool_op_map, NULL);

    return 0;
}


int unregister_tm2_l2pool_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_L2POOL, 1, tm2_load_l2pool);

    return 0;
}
