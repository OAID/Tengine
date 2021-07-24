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

#include "softmax_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/sys_port.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* ir_graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    ir_tensor_t* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    output->layout = input->layout;

    set_ir_tensor_shape(output, input->dims, input->dim_num);

    return 0;
}


static int init_op(ir_op_t* op)
{
    struct softmax_param* softmax_param = ( struct softmax_param* )sys_malloc(sizeof(struct softmax_param));

    if (softmax_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    softmax_param->axis = 1;

    op->param_mem = softmax_param;
    op->param_size = sizeof(struct softmax_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_softmax_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SOFTMAX, OP_SOFTMAX_NAME, &m);
}


int unregister_softmax_op()
{
    return unregister_op(OP_SOFTMAX, 1);
}
