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

#include "priorbox_param.h"

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


static int priorbox_op_map(int op)
{
    return OP_PRIORBOX;
}


static int tm2_load_priorbox(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node, const TM2_Operator* tm_op)
{
    struct priorbox_param* priorbox_param = (struct priorbox_param*)ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_PriorBoxParam* tm_param = (TM2_PriorBoxParam*)(mem_base + tm_op->offset_t_param);
    const TM2_Vector_floats* v_minsizes = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_min_size);
    const TM2_Vector_floats* v_maxsizes = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_max_size);
    const TM2_Vector_floats* v_variances = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_variance);
    const TM2_Vector_floats* v_ratios = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_aspect_ratio);

    priorbox_param->min_size_num = v_minsizes->v_num;
    priorbox_param->min_size = (float*)sys_malloc(v_minsizes->v_num * sizeof(float));
    for (unsigned int i = 0; i < v_minsizes->v_num; i++)
        priorbox_param->min_size[i] = v_minsizes->data[i];

    priorbox_param->max_size_num = v_maxsizes->v_num;
    priorbox_param->max_size = (float*)sys_malloc(v_maxsizes->v_num * sizeof(float));
    for (unsigned int i = 0; i < v_maxsizes->v_num; i++)
        priorbox_param->max_size[i] = v_maxsizes->data[i];

    priorbox_param->variance = (float*)sys_malloc(v_variances->v_num * sizeof(float));
    for (unsigned int i = 0; i < v_variances->v_num; i++)
        priorbox_param->variance[i] = v_variances->data[i];

    priorbox_param->aspect_ratio_size = v_ratios->v_num;
    priorbox_param->aspect_ratio = (float*)sys_malloc(v_ratios->v_num * sizeof(float));
    for (unsigned int i = 0; i < v_ratios->v_num; i++)
        priorbox_param->aspect_ratio[i] = v_ratios->data[i];

    priorbox_param->clip = tm_param->clip;
    priorbox_param->flip = tm_param->flip;
    priorbox_param->image_h = tm_param->img_h;
    priorbox_param->image_size = tm_param->img_size;
    priorbox_param->image_w = tm_param->img_w;
    priorbox_param->num_priors = tm_param->num_priors;
    priorbox_param->offset = tm_param->offset;
    priorbox_param->out_dim = tm_param->out_dim;
    priorbox_param->step_h = tm_param->step_h;
    priorbox_param->step_w = tm_param->step_w;

    return 0;
}


// TODO: add unload op


int register_tm2_priorbox_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_PRIORBOX, 1, tm2_load_priorbox, priorbox_op_map, NULL);

    return 0;
}


int unregister_tm2_priorbox_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_PRIORBOX, 1, tm2_load_priorbox);

    return 0;
}
