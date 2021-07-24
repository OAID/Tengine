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

#include "unsqueeze_param.h"

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
                return -1;

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


static int init_op(struct op* op)
{
    struct unsqueeze_param* unsqueeze_param = ( struct unsqueeze_param* )sys_malloc(sizeof(struct unsqueeze_param));

    if (unsqueeze_param == NULL)
    {
        return -1;
    }

    unsqueeze_param->axises_size = 1;

    op->param_mem = unsqueeze_param;
    op->param_size = sizeof(struct unsqueeze_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct unsqueeze_param* unsqueeze_param = (struct unsqueeze_param*)op->param_mem;
    if (unsqueeze_param->axises)
        sys_free(unsqueeze_param->axises);
    sys_free(op->param_mem);
}


int register_unsqueeze_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_UNSQUEEZE, OP_UNSQUEEZE_NAME, &m);
}


int unregister_unsqueeze_op()
{
    return unregister_op(OP_UNSQUEEZE, 1);
}
