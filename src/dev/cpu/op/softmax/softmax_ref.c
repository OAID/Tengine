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

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "softmax_param.h"
#include "compiler_fp16.h"
#include <math.h>

/**
 * @brief softmax function
 * @param[in]       vec_in      pointer to input vector
 * @param[in]       dim_vec     input vector dimention
 * @param[out]      p_out       pointer to output vector
 * @return none.
 *
 */

static void GetMaxArray(void* input, void* array, int in_size, int on_size, int num_thread)
{
    float* input_ptr = ( float* )input;
    float* array_ptr = ( float* )array;

    memcpy(array_ptr, input_ptr, in_size * sizeof(float));

    // #pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            if (array_ptr[l] < input_ptr[j * in_size + l])
                array_ptr[l] = input_ptr[j * in_size + l];
        }
    }
}

static void GetOutResult(void* input, void* output, void* array, void* sum_array, int in_size, int on_size,
                         int num_thread)
{
    float* input_ptr = ( float* )input;
    float* output_ptr = ( float* )output;
    float* array_ptr = ( float* )array;
    float* sum_array_ptr = ( float* )sum_array;

    memset(sum_array, 0x0, in_size * sizeof(float));

    /* get the exp and the summary */
    // #pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
            sum_array_ptr[l] += output_ptr[index];
        }
    }

    /* the final result */
    // #pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < on_size; j++)
    {
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] /= sum_array_ptr[l];
        }
    }
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct softmax_param* softmax_param = ( struct softmax_param* )ir_node->op.param_mem;

    int element_size = input_tensor->elem_size;
    int type = input_tensor->data_type;

    int dims[4];
    for (int i = 0; i < input_tensor->dim_num; i++)
    {
        dims[i] = input_tensor->dims[i];
    }

    int axis = softmax_param->axis;
    int out_size, in_size, on_size;

    out_size = 1;
    for (int i = 0; i < axis; i++)
    {
        out_size *= dims[i];
    }

    in_size = 1;
    for (size_t i = axis + 1; i < input_tensor->dim_num; i++)
    {
        in_size *= dims[i];
    }
    on_size = dims[axis];

    float* max_array = ( float* )malloc(in_size * sizeof(float));
    float* sum_array = ( float* )malloc(in_size * sizeof(float));

    int on_in_size = on_size * in_size;

    if (type == TENGINE_DT_FP32)
    {
        float* input = input_tensor->data;
        float* output = output_tensor->data;

        for (int i = 0; i < out_size; i++)
        {
            /* get max */
            int img_base = i * on_in_size;

            GetMaxArray(input + img_base, max_array, in_size, on_size, exec_graph->num_thread);
            GetOutResult(input + img_base, output + img_base, max_array, sum_array, in_size, on_size,
                         exec_graph->num_thread);
        }
    }
    else if (type == TENGINE_DT_FP16)
    {
        int totol_size = on_in_size * out_size;
        __fp16* input = input_tensor->data;
        __fp16* output = output_tensor->data;
        float* input_f = ( float* )malloc(totol_size * 4);
        float* output_f = ( float* )malloc(totol_size * 4);

        /* fp16 to fp32 */
        for (int i = 0; i < out_size; i++)
            for (int j = 0; j < on_in_size; j++)
                input_f[i * on_in_size + j] = fp16_to_fp32(input[i * on_in_size + j]);

        /* fp32 softmax */
        for (int i = 0; i < out_size; i++)
        {
            /* get max */
            int img_base = i * in_size * on_size;
            GetMaxArray(input_f + img_base, max_array, in_size, on_size, exec_graph->num_thread);
            GetOutResult(input_f + img_base, output_f + img_base, max_array, sum_array, in_size, on_size,
                         exec_graph->num_thread);
        }

        /* fp32 to fp16 */
        for (int i = 0; i < out_size; i++)
            for (int j = 0; j < on_in_size; j++)
                output[i * on_in_size + j] = fp32_to_fp16(output_f[i * on_in_size + j]);

        free(input_f);
        free(output_f);
    }
    else if (type == TENGINE_DT_UINT8)
    {
        int totol_size = on_in_size * out_size;

        uint8_t* input = input_tensor->data;
        uint8_t* output = output_tensor->data;
        float* input_f = ( float* )malloc(totol_size * 4);
        float* output_f = ( float* )malloc(totol_size * 4);

        float input_scale = input_tensor->scale;
        float output_scale = output_tensor->scale;
        uint8_t input_zero = input_tensor->zero_point;
        uint8_t output_zero = output_tensor->zero_point;

        /* dequant to fp32 */
        for (int i = 0; i < out_size; i++)
            for (int j = 0; j < on_in_size; j++)
                input_f[i * on_in_size + j] = ((float)input[i * on_in_size + j] - (float)input_zero) * input_scale;

        /* fp32 softmax */
        for (int i = 0; i < out_size; i++)
        {
            /* get max */
            int img_base = i * in_size * on_size;
            GetMaxArray(input_f + img_base, max_array, in_size, on_size, exec_graph->num_thread);
            GetOutResult(input_f + img_base, output_f + img_base, max_array, sum_array, in_size, on_size,
                         exec_graph->num_thread);
        }

        /* quant to uint8 */
        for (int i = 0; i < out_size; i++)
        {
            for (int j = 0; j < on_in_size; j++)
            {
                int udata = (int)(round(output_f[i * on_in_size + j] / output_scale) + output_zero);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;
                output[i * on_in_size + j] = udata;
            }
        }

        free(input_f);
        free(output_f);
    }
    else
    {
        printf("Input data type %d not to be supported.\n", type);
        return -1;
    }

    free(max_array);
    free(sum_array);

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] ||
        input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_softmax_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}

static int unreg_softmax_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_softmax_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_softmax_hcl_ops);
