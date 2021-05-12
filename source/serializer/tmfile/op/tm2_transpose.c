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

#include "transpose_param.h"

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


static int transpose_op_map(int op)
{
    return OP_TRANSPOSE;
}


static int tm2_load_transpose(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node, const TM2_Operator* tm_op)
{
    struct transpose_param* transpose_param = ( struct transpose_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_TransposeParam* tm_param = (TM2_TransposeParam*)(mem_base + tm_op->offset_t_param);

    if (tm_param->offset_tr_shape != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_re_shape = (TM2_Vector_dims*)(mem_base + tm_param->offset_tr_shape);
        transpose_param->tr_shape = (int*)sys_malloc(v_re_shape->v_num * sizeof(int));
        transpose_param->tr_shape_size = v_re_shape->v_num;

        for (unsigned int i = 0; i < v_re_shape->v_num; i++)
        {
            transpose_param->tr_shape[i] = v_re_shape->dims[i];
        }
    }

    return 0;
}


int register_tm2_transpose_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_TRANSPOSE, 1, tm2_load_transpose, transpose_op_map, NULL);

    return 0;
}


int unregister_tm2_transpose_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_TRANSPOSE, 1, tm2_load_transpose);

    return 0;
}
