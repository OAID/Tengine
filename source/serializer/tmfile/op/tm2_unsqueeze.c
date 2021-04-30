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
 * Author: bhu@openailab.com
 */

#include "unsqueeze_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int unsqueeze_op_map(int op)
{
    return OP_UNSQUEEZE;
}


static int tm2_load_unsqueeze(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                              const TM2_Operator* tm_op)
{
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_UnsqueezeParam* tm_param = ( TM2_UnsqueezeParam* )(mem_base + tm_op->offset_t_param);

    if (tm_param->offset_vi_axises != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_axises = ( TM2_Vector_dims* )(mem_base + tm_param->offset_vi_axises);
        unsqueeze_param->axises_size = v_axises->v_num;
        unsqueeze_param->axises = ( int* )sys_malloc(v_axises->v_num * sizeof(int));
        for (unsigned int i = 0; i < v_axises->v_num; i++)
            unsqueeze_param->axises[i] = v_axises->dims[i];
    }

    return 0;
}


int register_tm2_unsqueeze_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_UNSQUEEZE, 1, tm2_load_unsqueeze, unsqueeze_op_map, NULL);

    return 0;
}


int unregister_tm2_unsqueeze_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_UNSQUEEZE, 1, tm2_load_unsqueeze);

    return 0;
}
