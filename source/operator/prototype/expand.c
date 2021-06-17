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

#include "expand_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/log.h"

#include <string.h>

static int infer_shape(struct node* node)
{
    struct vector* dims = create_vector(sizeof(int), NULL);
    struct vector* dims1 = create_vector(sizeof(int), NULL);
    struct vector* dims2 = create_vector(sizeof(int), NULL);
    
    expand_param_t* param = ( struct expand_param* )(node->op.param_mem);

    struct graph* graph = node->graph;
    struct tensor* input1 = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* input2 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int flag = 1;
    int32_t * input2_data = input2->data;
    for(int i = 0; i < input2->elem_num; i++)
    {
        if(input2_data[i] == 0){
            flag = 0;
        }
    }

    if(flag == 1)
    {
        for(int i = 0; i < input2->elem_num; i++)
            param->ex_shape[i] = input2_data[i];
    }
    
    for(int i = 0; i < (int)param->dim_num; i++)
    {
        int temp = param->ex_shape[i];
        push_vector_data(dims2, (void*)&temp);
    }
    int num = get_vector_num(dims2);


    int input1_dim_size = input1->dim_num;
    int input2_dim_size = param->dim_num;
    
    if(input1_dim_size == input2_dim_size)
    {
        for(int i = 0; i < input2_dim_size; i++)
        {
            if(input1->dims[i] >= param->ex_shape[i])
            {
                int temp = input1->dims[i];
                push_vector_data(dims, (void*)&temp);
            } 
            else
            {
                int temp = param->ex_shape[i];
                push_vector_data(dims, (void*)&temp);
            }
        }        
    } else {
        int diff = fabs(input1_dim_size - input2_dim_size);
        if(input1_dim_size > input2_dim_size)
        {
            for(int i = 0; i < input1_dim_size; i++)
            {
                int temp = input1->dims[i];
                push_vector_data(dims, (void*)&temp);
            }
            for(int i = 0; i < input1_dim_size - diff; i++)
            {
                if(input1->dims[i+diff] > param->ex_shape[i])
                {
                    int temp = input1->dims[i+diff];
                    push_vector_data(dims, (void*)&temp);
                } 
                else 
                {
                    int temp = param->ex_shape[i];
                    push_vector_data(dims, (void*)&temp);
                }                
            }
        } else {
            for(int i = 0; i < input2_dim_size; i++)
            {
                int temp = param->ex_shape[i];
                push_vector_data(dims, (void*)&temp);
            }
            for(int i = 0; i < input2_dim_size - diff; i++)
            {
                if(param->ex_shape[i+diff] > input1->dims[i])
                {
                    int temp = param->ex_shape[i+diff];
                    push_vector_data(dims, (void*)&temp);
                } 
                else 
                {
                    int temp = input1->dims[i];
                    push_vector_data(dims, (void*)&temp);
                }                
            }
        }
    }
    int new_size = 1;
    int* new_shape_temp = (int*)sys_malloc(get_vector_num(dims)*sizeof(int));
    for(int i = 0; i < get_vector_num(dims); i++)
    {
        int* a = (int*)get_vector_data(dims, i);
        new_shape_temp[i] = *a;
    }

    output->layout = input1->layout;
    int ret = set_ir_tensor_shape(output, new_shape_temp, get_vector_num(dims));

    sys_free(new_shape_temp);
    release_vector(dims);
    release_vector(dims1);
    release_vector(dims2);
    return ret;
}


static int init_op(struct op* op)
{
    struct expand_param* expand_param = ( struct expand_param* )sys_malloc(sizeof(struct expand_param));

    if (expand_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    memset(expand_param, 0, sizeof(struct expand_param));
    op->param_mem = expand_param;
    op->param_size = sizeof(struct expand_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct expand_param* expand_param = ( struct expand_param* )op->param_mem;

    if (expand_param->ex_shape)
        sys_free(expand_param->ex_shape);

    sys_free(op->param_mem);
}


int register_expand_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_EXPAND, OP_EXPAND_NAME, &m);
}


int unregister_expand_op()
{
    return unregister_op(OP_EXPAND, 1);
}
