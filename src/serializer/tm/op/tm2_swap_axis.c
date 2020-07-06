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
 * Author: haitao@openailab.com
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
#include "swap_axis_param.h"

static int swap_axis_op_map(int op)
{
    return OP_SWAP_AXIS;
}

static int tm2_load_swap_axis(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                              const TM2_Operator* tm_op)
{
    struct swap_axis_param* swap_axis_param = ( struct swap_axis_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_SwapAxisParam* tm_param = ( TM2_SwapAxisParam* )(mem_base + tm_op->offset_t_param);

    swap_axis_param->dim_0 = tm_param->dim_0;
    swap_axis_param->dim_1 = tm_param->dim_1;

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

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_SWAPAXIS, 1, tm2_load_swap_axis, swap_axis_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_SWAPAXIS, 1, tm2_load_swap_axis);

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_swap_axis_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_swap_axis_ops", unreg_tm2_ops);
