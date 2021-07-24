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
 * Author: chh@openailab.com
 */

#include "crop_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/log.h"


static int crop_op_map(int op)
{
    return OP_CROP;
}


static int tm2_load_crop(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                         const TM2_Operator* tm_op)
{
    struct crop_param* crop_param = ( struct crop_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_CropParam* tm_param = ( TM2_CropParam* )(mem_base + tm_op->offset_t_param);

    crop_param->num_args = tm_param->num_args;
    crop_param->offset_c = tm_param->offset_c;
    crop_param->offset_h = tm_param->offset_h;
    crop_param->offset_w = tm_param->offset_w;
    crop_param->crop_h = tm_param->crop_h;
    crop_param->crop_w = tm_param->crop_w;
    crop_param->center_crop = tm_param->center_crop;
    crop_param->axis = tm_param->axis;
    crop_param->flag = tm_param->flag;

    return 0;
}


int register_tm2_crop_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_CROP, 1, tm2_load_crop, crop_op_map, NULL);

    return 0;
}


int unregister_tm2_crop_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_CROP, 1, tm2_load_crop);

    return 0;
}
