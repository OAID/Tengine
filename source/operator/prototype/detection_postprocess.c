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

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input0 = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* input1 = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);

    struct tensor* output0 = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct tensor* output1 = get_ir_graph_tensor(ir_graph, node->output_tensors[1]);
    struct tensor* output2 = get_ir_graph_tensor(ir_graph, node->output_tensors[2]);
    struct tensor* output3 = get_ir_graph_tensor(ir_graph, node->output_tensors[3]);

    struct detection_postprocess_param* detection_postprocess_param =
        ( struct detection_postprocess_param* )(node->op.param_mem);
    int max_detections = detection_postprocess_param->max_detections;
    int max_classes_per_detection = detection_postprocess_param->max_classes_per_detection;
    int num_classes = detection_postprocess_param->num_classes;
    int num_detected_boxes = max_detections * max_classes_per_detection;
    int* in_dim1 = &input0->dims[TE_MAX_SHAPE_DIM_NUM];
    int* in_dim2 = &input1->dims[TE_MAX_SHAPE_DIM_NUM];

    // Only support: batch_size == 1 && num_coord == 4
    if (input0->dims[0] != 1 || input0->dims[1] != 4 || input1->dims[0] != 1 || input1->dims[2] != input0->dims[2] ||
        input1->dims[1] != num_classes + 1)
    {
        TLOG_ERR("Not Support.\n");
        return -1;
    }
    int dim0[4] = {1, 4, num_detected_boxes};
    int dim1[2] = {1, num_detected_boxes};
    int dim2[2] = {1, num_detected_boxes};
    int dim3[1] = {1};

    set_ir_tensor_shape(output0, dim0, 3);
    set_ir_tensor_shape(output1, dim1, 2);
    set_ir_tensor_shape(output2, dim2, 2);
    set_ir_tensor_shape(output3, dim3, 1);

    return 0;
}


static int init_op(struct op* op)
{
    struct detection_postprocess_param* detection_postprocess_param =
        ( struct detection_postprocess_param* )sys_malloc(sizeof(struct detection_postprocess_param));

    if (detection_postprocess_param == NULL)
    {
        return -1;
    }

    detection_postprocess_param->scales = NULL;

    op->param_mem = detection_postprocess_param;
    op->param_size = sizeof(struct detection_postprocess_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct detection_postprocess_param* detection_postprocess_param =
        ( struct detection_postprocess_param* )op->param_mem;

    if (detection_postprocess_param->scales)
        sys_free(detection_postprocess_param->scales);

    sys_free(op->param_mem);
}


int register_detection_postprocess_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_DETECTION_POSTPROCESS, OP_DETECTION_POSTPROCESS_NAME, &m);
}


int unregister_detection_postprocess_op()
{
    return unregister_op(OP_DETECTION_POSTPROCESS, 1);
}
