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

#include "detection_output_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct detection_output_param* param = ( struct detection_output_param* )node->op.param_mem;

    int dims[TE_MAX_SHAPE_DIM_NUM] = {0};

    dims[0] = input->dims[0];
    dims[1] = param->keep_top_k;
    dims[2] = 6;
    dims[3] = 1;

    output->layout = TENGINE_LAYOUT_NHWC;

    set_ir_tensor_shape(output, dims, 4);
    return 0;
}


static int init_op(struct op* op)
{
    struct detection_output_param* detection_output_param =
        ( struct detection_output_param* )sys_malloc(sizeof(struct detection_output_param));

    if (detection_output_param == NULL)
    {
        return -1;
    }

    detection_output_param->num_classes = 21;
    detection_output_param->keep_top_k = 100;
    detection_output_param->nms_top_k = 100;
    detection_output_param->confidence_threshold = 0.25f;
    detection_output_param->nms_threshold = 0.45f;

    op->param_mem = detection_output_param;
    op->param_size = sizeof(struct detection_output_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_detection_output_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_DETECTION_OUTPUT, OP_DETECTION_OUTPUT_NAME, &m);
}


int unregister_detection_output_op()
{
    return unregister_op(OP_DETECTION_OUTPUT, 1);
}
