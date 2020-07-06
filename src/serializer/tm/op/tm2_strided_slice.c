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
 * Author: sqfu@openailab.com
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
#include "strided_slice_param.h"

static int strided_slice_op_map(int op)
{
    return OP_STRIDED_SLICE;
}

static int tm2_load_strided_slice(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                                  const TM2_Operator* tm_op)
{
    struct strided_slice_param* strided_slice_param = ( struct strided_slice_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_StridedSliceParam* tm_param = ( TM2_StridedSliceParam* )(mem_base + tm_op->offset_t_param);

    strided_slice_param->begin[0] = tm_param->begin_n;
    strided_slice_param->begin[1] = tm_param->begin_c;
    strided_slice_param->begin[2] = tm_param->begin_h;
    strided_slice_param->begin[3] = tm_param->begin_w;
    strided_slice_param->end[0] = tm_param->end_n;
    strided_slice_param->end[1] = tm_param->end_c;
    strided_slice_param->end[2] = tm_param->end_h;
    strided_slice_param->end[3] = tm_param->end_w;
    strided_slice_param->stride[0] = tm_param->stride_n;
    strided_slice_param->stride[1] = tm_param->stride_c;
    strided_slice_param->stride[2] = tm_param->stride_h;
    strided_slice_param->stride[3] = tm_param->stride_w;

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

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_STRIDEDSLICE, 1, tm2_load_strided_slice, strided_slice_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_STRIDEDSLICE, 1, tm2_load_strided_slice);
    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_strided_slice_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_strided_slice_ops", unreg_tm2_ops);
