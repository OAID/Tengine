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

#include "fc_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/log.h"


static int infer_shape(ir_node_t* node)
{
    ir_graph_t* graph = node->graph;
    ir_tensor_t* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    ir_tensor_t* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    ir_tensor_t* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dim[4];

    int n = weight->dims[0];
    int k = weight->dims[1];

    int m = input->dims[0];
    int input_k = input->dims[1];

    if (input->dim_num == 2)
    {
        dim[0] = m;
        dim[1] = n;
    }
    else if (input->dim_num == 3)
    {
        if (input->dims[2] != 0)
            input_k *= input->dims[2];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
        }
    }
    else if (input->dim_num == 4)
    {
        if (input->dims[2] * input->dims[3] != 0)
            input_k *= input->dims[2] * input->dims[3];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = 1;
            dim[3] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
            dim[3] = 1;
        }
    }
    else
        return -1;

    if (k != input_k)
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", k);
        return -1;
    }

    set_ir_tensor_shape(output, dim, input->dim_num);

    return 0;
}

static int init_op(ir_op_t* op)
{
    struct fc_param* fc_param = ( struct fc_param* )sys_malloc(sizeof(struct fc_param));

    if (fc_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    fc_param->num_output = 1;

    op->param_mem = fc_param;
    op->param_size = sizeof(struct fc_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(ir_op_t* op)
{
    sys_free(op->param_mem);
}

int register_fc_op()
{
    ir_method_t m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_FC, OP_FC_NAME, &m);
}

int unregister_fc_op()
{
    return unregister_op(OP_FC, 1);
}
