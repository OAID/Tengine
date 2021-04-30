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

#include "tile_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"
#include "utility/vector.h"

#include <string.h>


static int infer_shape(struct node* node)
{
    struct tile_param* param = (struct tile_param*)node->op.param_mem;

    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int frame = param->frame_flag;

    int output_n = 0;
    int output_c = 0;
    int output_h = 0;
    int output_w = 0;

    struct vector* reps_vector = create_vector(sizeof(int), NULL);

    for(int i = 0; i < param->reps_size; i++)
    {
        push_vector_data(reps_vector, (void*)&param->reps[i]);
    }

    if(frame == 0) // caffe
    {
        int param_size = get_vector_num(reps_vector);
        if(param_size != 0)
        {
            for(int i = 0; i < param_size / 2; i++)
            {
                int temp = ((int*)get_vector_data(reps_vector,0))[0];
                int ori_reps = ((int*)get_vector_data(reps_vector, param_size -i -1))[0];
                set_vector_data(reps_vector, i, (void*)&ori_reps);
            }
        }
        else
        {
            return -1;
        }
        int push_data = 1;
        switch(param_size)
        {
            case 0:
                for(int i = 0; i < 4; i++)
                {
                    push_vector_data(reps_vector, (void*)&push_data);
                }
                break;
            case 1:
                for(int i = 0; i < 3; i++)
                {
                    push_vector_data(reps_vector, (void*)&push_data);
                };
                break;
            case 2:
                for(int i = 0; i < 2; i++)
                {
                    push_vector_data(reps_vector, (void*)&push_data);
                }
                break;
            case 3:
                push_vector_data(reps_vector, (void*)&push_data);
                break;
            default:
                break;
        }

        output_n = input_tensor->dims[0]*(( int* )get_vector_data(reps_vector, 3))[0];
        output_c = input_tensor->dims[1]*(( int* )get_vector_data(reps_vector, 2))[0];
        output_h = input_tensor->dims[2]*(( int* )get_vector_data(reps_vector, 1))[0];
        output_w = input_tensor->dims[3]*(( int* )get_vector_data(reps_vector, 0))[0];
    } 
    else if (frame == 1) 
    {
        printf("Tile::InferShape onnx\n");
    }

    int* new_shape = (int*)sys_malloc(get_vector_num(reps_vector)*sizeof(int));
    for(int i = 0; i < get_vector_num(reps_vector); i++)
    {
        int* a = (int*)get_vector_data(reps_vector, i);
        new_shape[i] = *a;
    }

    set_ir_tensor_shape(output_tensor, new_shape, get_vector_num(reps_vector));
    sys_free(new_shape);
    release_vector(reps_vector);
    return 0;
}


static int init_op(struct op* op)
{
    struct tile_param* tile_param = ( struct tile_param* )sys_malloc(sizeof(struct tile_param));

    if (tile_param == NULL)
    {
        return -1;
    }

    memset(tile_param,0,sizeof(struct tile_param));
    op->param_mem = tile_param;
    op->param_size = sizeof(struct tile_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    struct tile_param* tile_param = (struct tile_param*)op->param_mem;
    if(tile_param->reps)
        sys_free(tile_param->reps);
    sys_free(op->param_mem);
}


int register_tile_op()
{
    struct method m;
    m.version = 1;
    m.init = init_op;
    m.release = release_op;


    return register_op(OP_TILE, OP_TILE_NAME, &m);

}


int unregister_tile_op()
{
    return unregister_op(OP_TILE,1);
}
