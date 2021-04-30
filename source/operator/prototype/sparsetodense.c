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

#include "sparsetodense_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"


static int infer_shape(struct node* node)
{
    struct sparsetodense_param* sparsetodense_param = ( struct sparsetodense_param* )(node->op.param_mem);

    struct graph* graph = node->graph;
    struct tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    if (input1->dim_num > 2)
        return -1;

    if (input1->dim_num == 2 && input0->dim_num == 2 && sparsetodense_param->output_shape_size1 != 0)
    {
        int dims[2];
        dims[0] = sparsetodense_param->output_shape_size0;
        dims[1] = sparsetodense_param->output_shape_size1;
        set_ir_tensor_shape(output, dims, 2);

        return 0;
    }
    else if (input1->dim_num == 1 && (input0->dim_num == 1 || input0->dim_num == 0))
    {
        int dims[1];
        dims[0] = sparsetodense_param->output_shape_size0;
        set_ir_tensor_shape(output, dims, 1);

        return 0;
    }
    else
    {
        return -1;
    }
}


static int init_op(struct op* op)
{
    struct sparsetodense_param* sparsetodense_param = (struct sparsetodense_param*)sys_malloc(sizeof(struct sparsetodense_param));

    if (sparsetodense_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    sparsetodense_param->default_value = 0;
    sparsetodense_param->output_shape_size0 = 1;
    sparsetodense_param->output_shape_size1 = 0;

    op->param_mem = sparsetodense_param;
    op->param_size = sizeof(struct sparsetodense_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_sparsetodense_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_SPARSETODENSE, OP_SPARSETODENSE_NAME, &m);
}


int unregister_sparsetodense_op()
{
    return unregister_op(OP_SPARSETODENSE, 1);
}
