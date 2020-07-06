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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "sparsetodense_param.h"

DEFINE_PARM_PARSE_ENTRY(sparsetodense_param, output_shape_size0, output_shape_size1, default_value);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct ir_tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct sparsetodense_param* sparsetodense_param = ( struct sparsetodense_param* )(node->op.param_mem);

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

static int init_op(struct ir_op* op)
{
    struct sparsetodense_param* sparsetodense_param =
        ( struct sparsetodense_param* )sys_malloc(sizeof(struct sparsetodense_param));

    if (sparsetodense_param == NULL)
    {
        set_tengine_errno(ENOMEM);
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

static void release_op(struct ir_op* op)
{
    sys_free(op->param_mem);
}

static int register_sparsetodense_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_SPARSETODENSE, OP_SPARSETODENSE_NAME, &m);
}

static int unregister_sparsetodense_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(sparsetodense_param));
    return unregister_op(OP_SPARSETODENSE, 1);
}

AUTO_REGISTER_OP(register_sparsetodense_op);
AUTO_UNREGISTER_OP(unregister_sparsetodense_op);
