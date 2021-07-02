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

#include "expand_param.h"

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


static int expand_op_map(int op)
{
    return OP_EXPAND;
}


static int tm2_load_expand(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct expand_param* param = ( struct expand_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_ExpandParam* tm_param = ( TM2_ExpandParam* )(mem_base + tm_op->offset_t_param);
    if (tm_param->offset_ex_shape != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_ex_shape = ( TM2_Vector_dims* )(mem_base + tm_param->offset_ex_shape);
        param->ex_shape = ( int* )sys_malloc(v_ex_shape->v_num * sizeof(int));

        for (unsigned int i = 0; i < v_ex_shape->v_num; i++)
        {
            param->ex_shape[i] = v_ex_shape->dims[i];
            
        }
    }
    param->dim_num = tm_param->dim_num;

    return 0;
}


int register_tm2_expand_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_EXPAND, 1, tm2_load_expand, expand_op_map, NULL);

    return 0;
}


int unregister_tm2_expand_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_EXPAND, 1, tm2_load_expand);

    return 0;
}
