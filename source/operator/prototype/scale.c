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

#include "scale_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* ir_graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    ir_tensor_t* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    set_ir_tensor_shape(output, input->dims, input->dim_num);

    return 0;
}


static int init_op(ir_op_t* op)
{
    struct scale_param* scale_param = ( struct scale_param* )sys_malloc(sizeof(struct scale_param));

    if (scale_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    scale_param->axis = 1;
    scale_param->num_axes = 1;
    scale_param->bias_term = 0;

    op->param_mem = scale_param;
    op->param_size = sizeof(struct scale_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}


int register_scale_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SCALE, OP_SCALE_NAME, &m);
}


int unregister_scale_op()
{
    return unregister_op(OP_SCALE, 1);
}
