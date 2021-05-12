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
 * Author: qli@openailab.com
 */

#include "detection_postprocess_param.h"

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


static int detection_postprocess_op_map(int op)
{
    return OP_DETECTION_POSTPROCESS;
}


static int tm2_load_detection_postprocess(struct graph* ir_graph, struct node* ir_node, const TM2_Node* tm_node,
                                          const TM2_Operator* tm_op)
{
    struct detection_postprocess_param* detection_postprocess_param =
        ( struct detection_postprocess_param* )ir_node->op.param_mem;
    const struct tm2_priv* tm2_priv = (struct tm2_priv*)ir_graph->serializer_privacy;
    const char* mem_base = tm2_priv->base;
    const TM2_DetectionPostProcessParam* tm_param =
        ( TM2_DetectionPostProcessParam* )(mem_base + tm_op->offset_t_param);

    detection_postprocess_param->max_detections = tm_param->max_detections;
    detection_postprocess_param->max_classes_per_detection = tm_param->max_classes_per_detection;
    detection_postprocess_param->nms_score_threshold = tm_param->nms_score_threshold;
    detection_postprocess_param->nms_iou_threshold = tm_param->nms_iou_threshold;
    detection_postprocess_param->num_classes = tm_param->num_classes;

    const TM2_Vector_floats* vf_scales = (TM2_Vector_floats*)(mem_base + tm_param->offset_vf_scales);
    detection_postprocess_param->scales = (float*)sys_malloc(vf_scales->v_num * sizeof(float));

    for (unsigned int i = 0; i < vf_scales->v_num;
         i++)    // TODO : need to check v_num .Next called in run function(detection_postprocess) default as 4 ?
        detection_postprocess_param->scales[i] = vf_scales->data[i];

    return 0;
}


int register_tm2_detection_postprocess_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    if (tm2_s == NULL)
    {
        TLOG_ERR("tengine serializer has not been registered yet\n");
        return -1;
    }

    tm2_s->register_op_loader(tm2_s, TM2_OPTYPE_DETECTIONPOSTPROCESS, 1, tm2_load_detection_postprocess,
                              detection_postprocess_op_map, NULL);

    return 0;
}


int unregister_tm2_detection_postprocess_op()
{
    struct serializer* tm2_s = find_serializer_via_name("tengine");

    tm2_s->unregister_op_loader(tm2_s, TM2_OPTYPE_DETECTIONPOSTPROCESS, 1, tm2_load_detection_postprocess);

    return 0;
}
