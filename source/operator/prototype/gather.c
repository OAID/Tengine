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
 * Author: zpluo@openailab.com
 */

#include "gather_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"


static int infer_shape(struct node* node)
{
    struct graph* graph   = node->graph;
    struct tensor* input  = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct gather_param* _param = ( struct gather_param* )(node->op.param_mem);
  
    int indices_size = _param->indices_num;
    
    struct vector* new_shape_temp = create_vector(sizeof(int), NULL);
    if(_param->is_onnx)
    {
        if(_param->axis == 0)
        {
            for(int i = 0; i < input->dim_num  - 1; i++)
            {
                push_vector_data(new_shape_temp, (void* )&input->dims[i+1]);
            }
        }
        else
        {
            for(int i = 0; i < input->dim_num; i++)
            {
                if(i == _param->axis)
                    push_vector_data(new_shape_temp, (void* )&indices_size);
                else
                    push_vector_data(new_shape_temp, (void* )&input->dims[i]);
            }
        }

        int* shape_temp = (int *)sys_malloc(get_vector_num(new_shape_temp) * sizeof(int));

        for (int i=0; i<get_vector_num(new_shape_temp); i++)
        {
            int* a = (int* )get_vector_data(new_shape_temp, i);
            shape_temp[i] = *a;
        }
        set_ir_tensor_shape(output, shape_temp, get_vector_num(new_shape_temp));
        sys_free(shape_temp);
    }
    else
    {
        int dims[4] ;
        dims[0] = input->dims[0];
        dims[1] = input->dims[1];
        dims[2] = input->dims[2];
        dims[3] = input->dims[3];

        if( _param->axis > ( int )input->dim_num) 
        {
            return -1;
        } 
        dims[_param->axis] = indices_size; 
        set_ir_tensor_shape(output, dims, 4);
    }
    
    release_vector(new_shape_temp);

    return 0;
}


static int init_op(struct op* op)
{
    struct gather_param* gather_param = ( struct gather_param* )sys_malloc(sizeof(struct gather_param));

    if (gather_param == NULL)
    {
        return -1;
    }

    /*set the param default value */
    gather_param->axis = 0;
    gather_param->indices_num = 0;

    op->param_mem = gather_param;
    op->param_size = sizeof(struct gather_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_gather_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_GATHER, OP_GATHER_NAME, &m);
}


int unregister_gather_op()
{
    return unregister_op(OP_GATHER, 1);
}
