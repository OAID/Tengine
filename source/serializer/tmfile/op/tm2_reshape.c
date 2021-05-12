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

#include "reshape_param.h"

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


static int reshape_op_map(int op)
{
    return OP_RESHAPE;
}


static int tm2_load_reshape(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct reshape_param* param = ( struct reshape_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_ReshapeParam* tm_param = ( TM2_ReshapeParam* )(mem_base + tm_op->offset_t_param);
    // set the reverse
    if (tm_param->reverse)
        param->reverse = true;
    else
        param->reverse = false;
    // set the is_mxnet
    if (tm_param->is_mxnet)
        param->is_mxnet = true;
    else
        param->is_mxnet = false;

    if (tm_param->offset_re_shape != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_re_shape = ( TM2_Vector_dims* )(mem_base + tm_param->offset_re_shape);
        param->dim_size = v_re_shape->v_num;

        param->re_shape = ( int* )sys_malloc(v_re_shape->v_num * sizeof(int));

        for (unsigned int i = 0; i < v_re_shape->v_num; i++)
        {
            param->re_shape[i] = v_re_shape->dims[i];
        }
    }

    return 0;
}


int register_tm2_reshape_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_RESHAPE, 1, tm2_load_reshape, reshape_op_map, NULL);

    return 0;
}


int unregister_tm2_reshape_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_RESHAPE, 1, tm2_load_reshape);

    return 0;
}
