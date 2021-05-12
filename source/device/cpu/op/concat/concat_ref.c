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
 * Author: jjzeng@openailab.com
 */

#include "concat_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <math.h>
#include <string.h>


struct shape_dim
{
    int dim[4];
    float scale;
    int zero;
};

struct concat_op_param
{
    struct shape_dim* input_shape;
    int input_counts;
    int input_dim;
    struct shape_dim output_shape;
    int output_dim;
    int axis;
    float out_scale;
    void** input_data;
};

static int ref_concat_fp32(const float** in_data, float* out_data, const struct concat_op_param* param)
{
    int axis = param->axis;
    int concat_dim = 0;
    for (int ii = 0; ii < param->input_counts; ++ii)
    {
        concat_dim += param->input_shape[ii].dim[axis];
    }

    if (concat_dim != param->output_shape.dim[axis])
    {
        TLOG_ERR("concant dimensions[%d] is not same output[%d]\n", concat_dim, param->output_shape.dim[axis]);
        return -1;
    }

    int out_size, in_size;

    out_size = 1;
    for (int ii = 0; ii < axis; ++ii)
    {
        out_size *= param->output_shape.dim[ii];
    }

    in_size = 1;
    for (int ii = axis + 1; ii < param->output_dim; ++ii)
    {
        in_size *= param->input_shape[0].dim[ii];
    }

    float* output_ptr = out_data;

    for (int k = 0; k < out_size; ++k)
    {
        //        #pragma omp parallel for num_threads(num_thread)
        for (int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            memcpy(output_ptr, in_data[j] + k * cp_size, cp_size * sizeof(float));
            output_ptr += cp_size;
        }
    }

    return 0;
}

static int ref_concat_fp16(const fp16_t** in_data, fp16_t* out_data, const struct concat_op_param* param)
{
    int axis = param->axis;
    int concat_dim = 0;
    for(int ii = 0; ii < param->input_counts; ++ii)
    {
        concat_dim += param->input_shape[ii].dim[axis];
    }

    if(concat_dim != param->output_shape.dim[axis])
    {
        TLOG_ERR("concat dimensions is not same output: ( %d -- %d )\n", concat_dim, param->output_shape.dim[axis]);
        return -1;
    }

    int out_size, in_size;

    out_size = 1;
    for(int ii = 0; ii < axis; ++ii)
    {
        out_size *= param->output_shape.dim[ii];
    }
    in_size = 1;
    for(int ii = axis + 1; ii < param->output_dim; ++ii)
    {
        in_size *= param->input_shape[0].dim[ii];
    }

    fp16_t* output_ptr = out_data;

    for(int k = 0; k < out_size; ++k)
    {
        for(int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            memcpy(output_ptr, in_data[j] + k * cp_size, cp_size * sizeof(fp16_t));
            output_ptr += cp_size;
        }
    }

    return 0;
}

static int ref_concat_uint8(const uint8_t** in_data, uint8_t* out_data, const struct concat_op_param* param)
{
    int axis = param->axis;
    int concat_dim = 0;
    for (int ii = 0; ii < param->input_counts; ++ii)
    {
        concat_dim += param->input_shape[ii].dim[axis];
    }

    if (concat_dim != param->output_shape.dim[axis])
    {
        TLOG_ERR("concat dimensions is not same output: ( %d -- %d )\n", concat_dim, param->output_shape.dim[axis]);
        return -1;
    }

    int outer_size, in_size;
    outer_size = 1;
    for (int ii = 0; ii < axis; ++ii)
    {
        outer_size *= param->output_shape.dim[ii];
    }
    in_size = 1;
    for (int ii = axis + 1; ii < param->output_dim; ++ii)
    {
        in_size *= param->output_shape.dim[ii];
    }

    int output_size = 1;
    for (int ii = 0; ii < param->output_dim; ++ii)
    {
        output_size *= param->output_shape.dim[ii];
    }

    uint8_t* output_ptr = out_data;
    float out_scale = param->output_shape.scale;
    uint8_t out_zero = param->output_shape.zero;
    for (int k = 0; k < outer_size; ++k)
    {
        for (int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            float scale = param->input_shape[j].scale;
            uint8_t input_zero = param->input_shape[j].zero;

            const uint8_t* input_ptr = ( const uint8_t* )(in_data[j] + k * cp_size);

            if (scale == out_scale && input_zero == out_zero)
            {
                memcpy(output_ptr, input_ptr, cp_size);
            }
            else
            {
                float t_scale = scale / out_scale;
                for (int ii = 0; ii < cp_size; ++ii)
                {
                    output_ptr[ii] = roundf((input_ptr[ii] - (float )input_zero) * t_scale) + (float )out_zero;
                }
            }
            output_ptr += cp_size;
        }
    }

    return 0;
}

static int ref_concat_int8(const int8_t** in_data, int8_t* out_data, const struct concat_op_param* param)
{
    int axis = param->axis;
    int concat_dim = 0;
    for (int ii = 0; ii < param->input_counts; ++ii)
    {
        concat_dim += param->input_shape[ii].dim[axis];
    }

    if (concat_dim != param->output_shape.dim[axis])
    {
        TLOG_ERR("concat dimensions is not same output: ( %d -- %d )\n", concat_dim, param->output_shape.dim[axis]);
        return -1;
    }

    int outer_size, in_size;
    outer_size = 1;
    for (int ii = 0; ii < axis; ++ii)
    {
        outer_size *= param->output_shape.dim[ii];
    }
    in_size = 1;
    for (int ii = axis + 1; ii < param->output_dim; ++ii)
    {
        in_size *= param->output_shape.dim[ii];
    }

    int output_size = 1;
    for (int ii = 0; ii < param->output_dim; ++ii)
    {
        output_size *= param->output_shape.dim[ii];
    }

    int8_t* output_ptr = out_data;
    float output_scale = param->output_shape.scale;
    for (int k = 0; k < outer_size; ++k)
    {
        for (int j = 0; j < param->input_counts; ++j)
        {
            int cp_size = param->input_shape[j].dim[axis] * in_size;
            float input_scale = param->input_shape[j].scale;

            const int8_t* input_ptr = ( const int8_t* )(in_data[j] + k * cp_size);

            if (input_scale == output_scale)
            {
                memcpy(output_ptr, input_ptr, cp_size);
            }
            else
            {
                float requant_scale = input_scale / output_scale;
                for (int ii = 0; ii < cp_size; ++ii)
                {
                    int data_i32 = round((float )input_ptr[ii] * requant_scale);
                    if (data_i32 > 127)
                        data_i32 = 127;
                    else if (data_i32 < -127)
                        data_i32 = -127;
                    output_ptr[ii] = (int8_t)data_i32;
                }
            }
            output_ptr += cp_size;
        }
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct concat_op_param* concat_op_param = ( struct concat_op_param* )sys_malloc(sizeof(struct concat_op_param));

    concat_op_param->axis = 0;
    concat_op_param->input_counts = 1;
    concat_op_param->input_dim = 1;
    concat_op_param->input_shape = NULL;
    concat_op_param->out_scale = 0.1f;
    concat_op_param->output_dim = 1;
    exec_node->ops_priv = concat_op_param;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* output_tensor;

    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct concat_op_param* concat_op_param = ( struct concat_op_param* )exec_node->ops_priv;
    struct concat_param* concat_param = ( struct concat_param* )ir_node->op.param_mem;

    concat_op_param->axis = concat_param->axis;
    concat_op_param->input_counts = ir_node->input_num;
    concat_op_param->input_shape = ( struct shape_dim* )sys_malloc(sizeof(struct shape_dim) * ir_node->input_num);

    concat_op_param->output_dim = output_tensor->dim_num;

    for (int ii = 0; ii < output_tensor->dim_num; ii++)
    {
        concat_op_param->output_shape.dim[ii] = output_tensor->dims[ii];
        concat_op_param->output_shape.scale = output_tensor->scale;
        concat_op_param->output_shape.zero = output_tensor->zero_point;
    }

    concat_op_param->input_data = ( void* )sys_malloc(sizeof(void*) * ir_node->input_num);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct concat_op_param* concat_op_param = ( struct concat_op_param* )exec_node->ops_priv;
    void* out_data = output_tensor->data;

    for (int i = 0; i < ir_node->input_num; i++)
    {
        input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        int number = input_tensor->dim_num;
        for (int j = 0; j < number; j++)
        {
            concat_op_param->input_shape[i].dim[j] = input_tensor->dims[j];
            concat_op_param->input_shape[i].scale = input_tensor->scale;
            concat_op_param->input_shape[i].zero = input_tensor->zero_point;
        }

        concat_op_param->input_data[i] = input_tensor->data;
    }

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_concat_fp32(( const float** )concat_op_param->input_data, out_data, concat_op_param);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
        ret = ref_concat_fp16(( const fp16_t** )concat_op_param->input_data, out_data, concat_op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_concat_uint8(( const uint8_t** )concat_op_param->input_data, out_data, concat_op_param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_concat_int8(( const int8_t** )concat_op_param->input_data, out_data, concat_op_param);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct concat_op_param* concat_op_param = ( struct concat_op_param* )exec_node->ops_priv;

    sys_free(concat_op_param->input_shape);
    sys_free(concat_op_param->input_data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
        .run = run,
        .reshape = NULL,
        .postrun = postrun,
        .init_node = init_node,
        .release_node = release_node,
        .score = score};

int register_concat_ref_op()
{
    return register_builtin_node_ops(OP_CONCAT, &hcl_node_ops);
}

int unregister_concat_ref_op()
{
    return unregister_builtin_node_ops(OP_CONCAT, &hcl_node_ops);
}
