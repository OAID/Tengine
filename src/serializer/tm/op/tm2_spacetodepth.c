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
 * Author: qtang@openailab.com
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

static int spacetodepth_op_map(int op)
{
    return OP_SPACETODEPTH;
}

static int tm2_load_spacetodepth(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                                 const TM2_Operator* tm_op)
{
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

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_SPACETODEPTH, 1, tm2_load_spacetodepth, spacetodepth_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_SPACETODEPTH, 1, tm2_load_spacetodepth);

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_spacetodepth_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_spacetodepth_ops", unreg_tm2_ops);
