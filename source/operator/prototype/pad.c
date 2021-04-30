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
 * Author: sqfu@openailab.com
 */

#include "pad_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct pad_param* pad_param = ( struct pad_param* )(node->op.param_mem);

    int dims[TE_MAX_SHAPE_DIM_NUM] = {0};
    if (pad_param->pad_0_h != -1 && pad_param->pad_0_w != -1 && pad_param->pad_1_h != -1 && pad_param->pad_1_w != -1 &&
        pad_param->pad_2_h != -1 && pad_param->pad_2_w != -1 && pad_param->pad_3_h != -1 && pad_param->pad_3_w != -1)
    {
        dims[0] = input->dims[0] + pad_param->pad_0_h + pad_param->pad_0_w;
        dims[1] = input->dims[1] + pad_param->pad_1_h + pad_param->pad_1_w;
        dims[2] = input->dims[2] + pad_param->pad_2_h + pad_param->pad_2_w;
        dims[3] = input->dims[3] + pad_param->pad_3_h + pad_param->pad_3_w;
    }
    else
    {
        return 0;
    }

    set_ir_tensor_shape(output, dims, input->dim_num);

    return 0;
}


static int init_op(ir_op_t* op)
{
    struct pad_param* pad_param = ( struct pad_param* )sys_malloc(sizeof(struct pad_param));

    if (pad_param == NULL)
    {
        return -1;
    }

    pad_param->mode = 0;
    pad_param->pad_0_h = -1;    // n
    pad_param->pad_0_w = -1;
    pad_param->pad_1_h = -1;    // c
    pad_param->pad_1_w = -1;
    pad_param->pad_2_h = -1;    // h
    pad_param->pad_2_w = -1;
    pad_param->pad_3_h = -1;    // w
    pad_param->pad_3_w = -1;
    pad_param->value = 0;

    /*set the param default value */
    op->param_mem = pad_param;
    op->param_size = sizeof(struct pad_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_pad_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_PAD, OP_PAD_NAME, &m);
}


int unregister_pad_op()
{
    return unregister_op(OP_PAD, 1);
}
