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

#include "resize_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct resize_param* resize_param = ( struct resize_param* )(node->op.param_mem);

    int dims[4];
    dims[0] = input->dims[0];
    if (graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        dims[1] = input->dims[1];
        dims[2] = ( int )(input->dims[2] * resize_param->scale_h);
        dims[3] = ( int )(input->dims[3] * resize_param->scale_w);
    }
    else if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        dims[1] = ( int )(input->dims[1] * resize_param->scale_h);
        dims[2] = ( int )(input->dims[2] * resize_param->scale_w);
        dims[3] = input->dims[3];
    }
    else
    {
        TLOG_ERR("resizeolution infer shape: unknown graph layout: %d\n", graph->graph_layout);
        return -1;
    }

    set_ir_tensor_shape(output, dims, 4);

    return 0;
}


static int init_op(struct op* op)
{
    struct resize_param* resize_param = ( struct resize_param* )sys_malloc(sizeof(struct resize_param));

    if (resize_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    resize_param->scale_h = 1.0;
    resize_param->scale_w = 1.0;
    resize_param->type = 0;

    op->param_mem = resize_param;
    op->param_size = sizeof(struct resize_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_resize_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_RESIZE, OP_RESIZE_NAME, &m);
}


int unregister_resize_op()
{
    return unregister_op(OP_RESIZE, 1);
}
