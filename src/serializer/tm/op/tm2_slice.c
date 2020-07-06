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
#include "slice_param.h"

static int slice_op_map(int op)
{
    return OP_SLICE;
}

static int tm2_load_slice(struct ir_graph* ir_graph, struct ir_node* ir_node, const TM2_Node* tm_node,
                          const TM2_Operator* tm_op)
{
    struct slice_param* slice_param = ( struct slice_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = ( struct tm2_priv* )ir_graph->serializer_priv;
    const char* mem_base = tm2_priv->base;
    const TM2_SliceParam* tm_param = ( TM2_SliceParam* )(mem_base + tm_op->offset_t_param);

    slice_param->axis = tm_param->axis;
    slice_param->begin = tm_param->begin;
    slice_param->end = tm_param->end;
    slice_param->iscaffe = tm_param->iscaffe;
    slice_param->ismxnet = tm_param->ismxnet;
    slice_param->isonnx = tm_param->isonnx;

    slice_param->begin_ = create_vector(sizeof(uint32_t), NULL);
    slice_param->size_ = create_vector(sizeof(uint32_t), NULL);
    slice_param->slice_point_ = create_vector(sizeof(uint32_t), NULL);

    if (tm_param->offset_vi_begins != TM2_NOT_SET)
    {
        const TM2_Vector_indices* v_begins = ( TM2_Vector_indices* )(mem_base + tm_param->offset_vi_begins);
        for (unsigned int i = 0; i < v_begins->v_num; i++)
        {
            push_vector_data(slice_param->begin_, ( void* )&v_begins->indices[i]);
        }
    }

    if (tm_param->offset_vi_sizes != TM2_NOT_SET)
    {
        const TM2_Vector_indices* v_size = ( TM2_Vector_indices* )(mem_base + tm_param->offset_vi_sizes);
        for (unsigned int i = 0; i < v_size->v_num; i++)
        {
            push_vector_data(slice_param->size_, ( void* )&v_size->indices[i]);
        }
    }

    if (tm_param->offset_vi_slice_points != TM2_NOT_SET)
    {
        const TM2_Vector_indices* v_slice_point = ( TM2_Vector_indices* )(mem_base + tm_param->offset_vi_slice_points);
        for (unsigned int i = 0; i < v_slice_point->v_num; i++)
        {
            push_vector_data(slice_param->slice_point_, ( void* )&v_slice_point->indices[i]);
        }
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

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_SLICE, 1, tm2_load_slice, slice_op_map, NULL);

    return 0;
}

static int unreg_tm2_ops(void* arg)
{
    struct serializer* tm2_s = find_serializer("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_SLICE, 1, tm2_load_slice);

    return 0;
}

REGISTER_MODULE_INIT(MOD_OP_LEVEL, "reg_slice_ops", reg_tm2_ops);
REGISTER_MODULE_EXIT(MOD_OP_LEVEL, "unreg_slice_ops", unreg_tm2_ops);
