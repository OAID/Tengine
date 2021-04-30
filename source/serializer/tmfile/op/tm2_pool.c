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

#include "pooling_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int pooling_op_map(int op)
{
    return OP_POOL;
}

static int tm2_load_pooling(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node, const TM2_Operator* tm_op)
{
    struct pool_param* pool_param = ( struct pool_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_PoolParam* tm_param = ( TM2_PoolParam* )(mem_base + tm_op->offset_t_param);

    pool_param->kernel_h = tm_param->kernel_h;
    pool_param->kernel_w = tm_param->kernel_w;
    pool_param->stride_h = tm_param->stride_h;
    pool_param->stride_w = tm_param->stride_w;
    pool_param->global = tm_param->global;
    pool_param->caffe_flavor = tm_param->caffe_flavor;

    pool_param->pad_h0 = tm_param->pad_h0;
    pool_param->pad_h1 = tm_param->pad_h1;
    pool_param->pad_w0 = tm_param->pad_w0;
    pool_param->pad_w1 = tm_param->pad_w1;

    pool_param->pad_h0_org = tm_param->pad_h0;
    pool_param->pad_h1_org = tm_param->pad_h1;
    pool_param->pad_w0_org = tm_param->pad_w0;
    pool_param->pad_w1_org = tm_param->pad_w1;

    pool_param->pool_method = tm_param->alg;

    return 0;
}


int register_tm2_pool_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_POOLING, 1, tm2_load_pooling, pooling_op_map, NULL);

    return 0;
}


int unregister_tm2_pool_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_POOLING, 1, tm2_load_pooling);

    return 0;
}
