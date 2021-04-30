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
 * Author: sqfu@openailab.com
 */

#include "strided_slice_param.h"

#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "utility/sys_port.h"

#include <math.h>


static int infer_shape(struct node* node)
{
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct strided_slice_param* param_ = ( struct strided_slice_param* )(node->op.param_mem);

    int delta_0 = (-param_->begin[0] + param_->end[0]) < 0 ? param_->begin[0] - param_->end[0] :
                                                             -param_->begin[0] + param_->end[0];
    int delta_1 = (-param_->begin[1] + param_->end[1]) < 0 ? param_->begin[1] - param_->end[1] :
                                                             -param_->begin[1] + param_->end[1];
    int delta_2 = (-param_->begin[2] + param_->end[2]) < 0 ? param_->begin[2] - param_->end[2] :
                                                             -param_->begin[2] + param_->end[2];
    int delta_3 = (-param_->begin[3] + param_->end[3]) < 0 ? param_->begin[3] - param_->end[3] :
                                                             -param_->begin[3] + param_->end[3];

    int dims[4] = {0};
    dims[0] = ceil((( float )input->dims[0] - ( float )delta_0) / ( float )param_->stride[0]);
    dims[1] = ceil((( float )input->dims[1] - ( float )delta_1) / ( float )param_->stride[1]);
    dims[2] = ceil((( float )input->dims[2] - ( float )delta_2) / ( float )param_->stride[2]);
    dims[3] = ceil((( float )input->dims[3] - ( float )delta_3) / ( float )param_->stride[3]);

    for (int i=0; i<4; i++)
    {
        if (dims[i] == 0)
            dims[i] = 1;
    }

    set_ir_tensor_shape(output, dims, input->dim_num);

    return 0;
}


static int init_op(struct op* op)
{
    struct strided_slice_param* strided_slice_param =
        ( struct strided_slice_param* )sys_malloc(sizeof(struct strided_slice_param));

    if (strided_slice_param == NULL)
    {
        return -1;
    }

    strided_slice_param->shrink_axis_mask = 0;
    strided_slice_param->new_axis_mask = 0;
    strided_slice_param->ellipsis_mask = 0;
    strided_slice_param->begin_mask = 0;
    strided_slice_param->end_mask = 0;

    /*set the param default value */
    op->param_mem = strided_slice_param;
    op->param_size = sizeof(struct strided_slice_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}


static void release_op(struct op* op)
{
    sys_free(op->param_mem);
}


int register_strided_slice_op()
{
    struct method m;

    m.version = 1;
    m.init = init_op;
    m.release = release_op;

    return register_op(OP_STRIDED_SLICE, OP_STRIDEDSLICE_NAME, &m);
}


int unregister_strided_slice_op()
{
    return unregister_op(OP_STRIDED_SLICE, 1);
}
