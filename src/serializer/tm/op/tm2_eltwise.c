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
#include "eltwise_param.h"

static int eltwise_op_map(int op)
{
    return OP_ELTWISE;
}

static int tm2_load_eltwise(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                            const TM2_Operator* tm_op)
{
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_EltwiseParam* tm_param = ( TM2_EltwiseParam* )(mem_base + tm_op->offset_t_param);

    eltwise_param->type = tm_param->type;
    eltwise_param->caffe_flavor = tm_param->caffe_flavor;
    eltwise_param->shift = tm_param->shift;
    eltwise_param->power = tm_param->power;
    eltwise_param->scale = tm_param->scale;

    return 0;
}

/* the auto register functions */

static int reg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_ELTWISE, 1, tm2_load_eltwise, eltwise_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_ELTWISE, 1, tm2_load_eltwise);

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_eltwise_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_eltwise_ops", unreg_tm2_ops);
