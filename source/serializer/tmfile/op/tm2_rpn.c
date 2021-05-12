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
 * Author: qtang@openailab.com
 */

#include "rpn_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "tmfile/tm2_serializer.h"
#include "device/device.h"
#include "utility/vector.h"
#include "utility/log.h"


static int rpn_op_map(int op)
{
    return OP_RPN;
}


static int tm2_load_rpn(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                        const TM2_Operator* tm_op)
{
    struct rpn_param* rpn_param = ( struct rpn_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_RPNParam* tm_param = ( TM2_RPNParam* )(mem_base + tm_op->offset_t_param);

    rpn_param->basesize = tm_param->basesize;
    rpn_param->feat_stride = tm_param->feat_stride;
    rpn_param->min_size = tm_param->min_size;
    rpn_param->nms_thresh = tm_param->nms_thresh;
    rpn_param->per_nms_topn = tm_param->per_nms_topn;
    rpn_param->post_nms_topn = tm_param->post_nms_topn;

    if (tm_param->offset_vf_anchor_scales != TM2_NOT_SET)
    {
        const TM2_Vector_floats* v_anchor_scales = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_anchor_scales);
        // param->dim_size = v_re_shape->v_num ;
        rpn_param->anchor_scales = create_vector(v_anchor_scales->v_num * sizeof(float), NULL);

        for (unsigned int i = 0; i < v_anchor_scales->v_num; i++)
        {
            push_vector_data(rpn_param->anchor_scales, ( void* )&v_anchor_scales->data[i]);
        }
    }

    if (tm_param->offset_vf_ratios != TM2_NOT_SET)
    {
        const TM2_Vector_floats* v_ratios = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_ratios);
        // param->dim_size = v_re_shape->v_num ;
        rpn_param->ratios = create_vector(v_ratios->v_num * sizeof(float), NULL);

        for (unsigned int i = 0; i < v_ratios->v_num; i++)
        {
            push_vector_data(rpn_param->ratios, ( void* )&v_ratios->data[i]);
        }
    }

    return 0;
}


int register_tm2_rpn_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_RPN, 1, tm2_load_rpn, rpn_op_map, NULL);

    return 0;
}


int unregister_tm2_rpn_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_RPN, 1, tm2_load_rpn);

    return 0;
}
