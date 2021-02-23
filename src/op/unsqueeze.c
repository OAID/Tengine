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
#include <stdbool.h>

#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "unsqueeze_param.h"

DEFINE_PARM_PARSE_ENTRY(unsqueeze_param, axises_size);

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )node->op.param_mem;

    int axises_size = unsqueeze_param->axises_size;
    int* out_dim = ( int* )sys_malloc((input->dim_num + axises_size) * sizeof(int));

    if (axises_size == 1)
    {
        for (int i = 0; i < input->dim_num; ++i)
        {
            out_dim[i] = input->dims[i];
        }

        for (unsigned int j = 0; j < unsqueeze_param->axises_size; j++)
        {
            int dim_size = input->dim_num;
            int pos = unsqueeze_param->axises[j];
            if (pos < 0)
            {
                pos = pos + dim_size;
            }

            if (pos < 0 || pos > dim_size)
                return false;

            for (int i = axises_size + input->dim_num - 1; i > pos; --i)
            {
                out_dim[i] = input->dims[i - 1];
            }

            out_dim[pos] = 1;
        }
    }
    else // test for OneFlow and daquanxian yyds
    {
        for (int i = 0; i < input->dim_num + axises_size; ++i)
        {
            out_dim[i] = -99;
        }

        for (unsigned int j = 0; j < unsqueeze_param->axises_size; j++)
        {
            int pos = unsqueeze_param->axises[j];
            out_dim[pos] = 1;
        }

        int k = 0;
        for (int i = 0; i < input->dim_num + axises_size; ++i)
        {
            if (out_dim[i] == -99)
            {
                out_dim[i] = input->dims[k];
                k++;
            }

        }
    }

    set_ir_tensor_shape(output, out_dim, axises_size + input->dim_num);

    sys_free(out_dim);

    return 0;
}

static int init_op(struct ir_op* op)
{
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )sys_malloc(sizeof(struct unsqueeze_param));

    if (unsqueeze_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    unsqueeze_param->axises_size = 1;

    op->param_mem = unsqueeze_param;
    op->param_size = sizeof(struct unsqueeze_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    struct unsqueeze_param* unsqueeze_param = (struct unsqueeze_param*)op->param_mem;
    if (unsqueeze_param->axises)
        sys_free(unsqueeze_param->axises);
    sys_free(op->param_mem);
}

static int register_unsqueeze_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_UNSQUEEZE, OP_UNSQUEEZE_NAME, &m);
}

static int unregister_unsqueeze_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(unsqueeze_param));
    return unregister_op(OP_UNSQUEEZE, 1);
}

AUTO_REGISTER_OP(register_unsqueeze_op);
AUTO_UNREGISTER_OP(unregister_unsqueeze_op);
