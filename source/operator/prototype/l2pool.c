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
 * Author: bzhang@openailab.com
 */

#include "l2pool_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct l2pool_param* l2pool_param = (struct l2pool_param* )(node->op.param_mem);

    int input_h = input_tensor->dims[1];
    int input_w = input_tensor->dims[2];
    int output_h = 0;
    int output_w = 0;

    if(l2pool_param->paddingType == 1){
        output_h = (input_h + l2pool_param->stride_h -1 )/l2pool_param->stride_h;
        output_w = (input_w + l2pool_param->stride_w -1 )/l2pool_param->stride_w;
    } else {
        output_h = (input_h + l2pool_param->stride_h - l2pool_param->kernel_h)/l2pool_param->stride_h;
        output_w = (input_w + l2pool_param->stride_w - l2pool_param->kernel_w)/l2pool_param->stride_w;
    }
    int dims[4];
    dims[0] = input_tensor->dims[0];
    dims[1] = output_h;
    dims[2] = output_w;
    dims[3] = input_tensor->dims[3];

    set_ir_tensor_shape(output_tensor, dims, 4);

    return 0;

}


static int init_op(struct op* op)
{
    struct l2pool_param* l2pool_param = ( struct l2pool_param* )sys_malloc(sizeof(struct l2pool_param));

    if (l2pool_param == NULL)
    {
        return -1;
    }

    l2pool_param->paddingType = 0;
    l2pool_param->kernel_h = 0;
    l2pool_param->kernel_w = 0;
    l2pool_param->stride_h = 0;
    l2pool_param->stride_w = 0;
    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_l2pool_op()
{
    struct method m;
    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_L2POOL, OP_L2POOL_NAME, &m);

}


int unregister_l2pool_op()
{
    return unregister_op(OP_L2POOL,1);
}
