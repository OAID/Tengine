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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: bhu@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_serializer.h"
#include "tm2_serializer.h"
#include "tengine_op.h"
#include "unsqueeze_param.h"

static int unsqueeze_op_map(int op)
{
    return OP_UNSQUEEZE;
}

static int tm2_load_unsqueeze(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                              const TM2_Operator* tm_op)
{
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
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

static int reg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_UNSQUEEZE, 1, tm2_load_unsqueeze, unsqueeze_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_UNSQUEEZE, 1, tm2_load_unsqueeze);

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_unsqueeze_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_unsqueeze_ops", unreg_tm2_ops);
