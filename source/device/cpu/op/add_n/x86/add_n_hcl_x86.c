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
 * Author: 412200533@qq.com
 */

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#if __SSE2__
#include <emmintrin.h>
#endif // __SSE2__
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#ifdef __TENGINE_OPENMP__
#include <omp.h>
#define TOMP(t) #pragma omp parallel for num_threads(t)
#else
#define TOMP(t)
#endif

struct add_n_op_param
{
    int in_num;
    void** input_data;
};

static int ref_add_n_fp32(const float** input, float* output, int size, const struct add_n_op_param* param, int num_thread)
{
    int in_num = param->in_num;
    #if __AVX__
    int count = size % 64;
    int sse_size = size - count;
    TOMP(num_thread)
    for (int i = 0; i < sse_size; i += 64)
    {
        __m256 _sum0 = _mm256_set1_ps(input[0]+i);
        __m256 _sum1 = _mm256_set1_ps(input[0]+i+8);
        __m256 _sum2 = _mm256_set1_ps(input[0]+i+16);
        __m256 _sum3 = _mm256_set1_ps(input[0]+i+24);
        __m256 _sum4 = _mm256_set1_ps(input[0]+i+32);
        __m256 _sum5 = _mm256_set1_ps(input[0]+i+40);
        __m256 _sum6 = _mm256_set1_ps(input[0]+i+48);
        __m256 _sum7 = _mm256_set1_ps(input[0]+i+56);
        float* output0 = output + i;
        float* output1 = output + i + 8;
        float* output2 = output + i + 16;
        float* output3 = output + i + 24;
        float* output4 = output + i + 32;
        float* output5 = output + i + 40;
        float* output6 = output + i + 48;
        float* output7 = output + i + 56;
        for (int n = 1; n < in_num; n++)
        {
            __m256 _op0 = _mm256_set1_ps(input[n]+i);
            __m256 _op1 = _mm256_set1_ps(input[n]+i+8);
            __m256 _op2 = _mm256_set1_ps(input[n]+i+16);
            __m256 _op3 = _mm256_set1_ps(input[n]+i+24);
            __m256 _op4 = _mm256_set1_ps(input[n]+i+32);
            __m256 _op5 = _mm256_set1_ps(input[n]+i+40);
            __m256 _op6 = _mm256_set1_ps(input[n]+i+48);
            __m256 _op7 = _mm256_set1_ps(input[n]+i+56);
            _sum0 = _mm256_add_ps(_sum0,_op0);
            _sum1 = _mm256_add_ps(_sum1,_op1);
            _sum2 = _mm256_add_ps(_sum2,_op2);
            _sum3 = _mm256_add_ps(_sum3,_op3);
            _sum4 = _mm256_add_ps(_sum4,_op4);
            _sum5 = _mm256_add_ps(_sum5,_op5);
            _sum6 = _mm256_add_ps(_sum6,_op6);
            _sum7 = _mm256_add_ps(_sum7,_op7);
            _mm256_store_ps(output0,_sum0);
            _mm256_store_ps(output1,_sum1);
            _mm256_store_ps(output2,_sum2);
            _mm256_store_ps(output3,_sum3);
            _mm256_store_ps(output4,_sum4);
            _mm256_store_ps(output5,_sum5);
            _mm256_store_ps(output6,_sum6);
            _mm256_store_ps(output7,_sum7);            
        }
    }
    TOMP(num_thread)
    for (int i = sse_size; i < size; i++)
    {
        output[i] = input[0][i];
        for (int n = 1; n < in_num; n++)
        {
            output[i] += input[n][i];
        }
    }
    #elif __SSE__
    int count = size % 32;
    int sse_size = size - count;
    TOMP(num_thread)
    for (int i = 0; i < sse_size; i += 32)
    {
        __m128 _sum0 = _mm_set1_ps(input[0]+i);
        __m128 _sum1 = _mm_set1_ps(input[0]+i+4);
        __m128 _sum2 = _mm_set1_ps(input[0]+i+8);
        __m128 _sum3 = _mm_set1_ps(input[0]+i+12);
        __m128 _sum4 = _mm_set1_ps(input[0]+i+16);
        __m128 _sum5 = _mm_set1_ps(input[0]+i+20);
        __m128 _sum6 = _mm_set1_ps(input[0]+i+24);
        __m128 _sum7 = _mm_set1_ps(input[0]+i+28);
        float* output0 = output + i;
        float* output1 = output + i + 4;
        float* output2 = output + i + 8;
        float* output3 = output + i + 12;
        float* output4 = output + i + 16;
        float* output5 = output + i + 20;
        float* output6 = output + i + 24;
        float* output7 = output + i + 28;
        for (int n = 1; n < in_num; n++)
        {
            __m128 _op0 = _mm_set1_ps(input[n]+i);
            __m128 _op1 = _mm_set1_ps(input[n]+i+4);
            __m128 _op2 = _mm_set1_ps(input[n]+i+8);
            __m128 _op3 = _mm_set1_ps(input[n]+i+12);
            __m128 _op4 = _mm_set1_ps(input[n]+i+16);
            __m128 _op5 = _mm_set1_ps(input[n]+i+20);
            __m128 _op6 = _mm_set1_ps(input[n]+i+24);
            __m128 _op7 = _mm_set1_ps(input[n]+i+28);
            _sum0 = _mm_add_ps(_sum0,_op0);
            _sum1 = _mm_add_ps(_sum1,_op1);
            _sum2 = _mm_add_ps(_sum2,_op2);
            _sum3 = _mm_add_ps(_sum3,_op3);
            _sum4 = _mm_add_ps(_sum4,_op4);
            _sum5 = _mm_add_ps(_sum5,_op5);
            _sum6 = _mm_add_ps(_sum6,_op6);
            _sum7 = _mm_add_ps(_sum7,_op7);
            _mm_store1_ps(output0,_sum0);
            _mm_store1_ps(output1,_sum1);
            _mm_store1_ps(output2,_sum2);
            _mm_store1_ps(output3,_sum3);
            _mm_store1_ps(output4,_sum4);
            _mm_store1_ps(output5,_sum5);
            _mm_store1_ps(output6,_sum6);
            _mm_store1_ps(output7,_sum7);            
        }
    }
    TOMP(num_thread)
    for (int i = sse_size; i < size; i++)
    {
        output[i] = input[0][i];
        for (int n = 1; n < in_num; n++)
        {
            output[i] += input[n][i];
        }
    }
    #else
    TOMP(num_thread)
    for (int i = 0; i < size; ++i)
    {
        output[i] = input[0][i];
        for (int n = 1; n < in_num; n++)
        {
            output[i] += input[n][i];
        }
    }
    #endif
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = ( struct add_n_op_param* )sys_malloc(sizeof(struct add_n_op_param));
    exec_node->ops_priv = add_n_op_param;

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
    struct add_n_op_param* add_n_op_param = ( struct add_n_op_param* )exec_node->ops_priv;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    int in_num = ir_node->input_num;
    add_n_op_param->in_num = in_num;
    add_n_op_param->input_data = ( void* )sys_malloc(sizeof(void*) * in_num);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor_a = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    uint32_t elem_num = input_tensor_a->elem_num;
    struct add_n_op_param* add_n_op_param = ( struct add_n_op_param* )exec_node->ops_priv;
    for (int i = 0; i < add_n_op_param->in_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        void* data = input_tensor->data;
        add_n_op_param->input_data[i] = data;
    }
    const void** input = ( const void** )add_n_op_param->input_data;

    float* output = output_tensor->data;
    for (uint32_t i = 0; i < elem_num; i++)
    {
        output[i] = 0;
    }
    ref_add_n_fp32(( const float** )input, output, elem_num, add_n_op_param, exec_graph->num_thread);
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = ( struct add_n_op_param* )exec_node->ops_priv;
    sys_free(add_n_op_param->input_data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops add_n_node_ops = {.prerun = prerun,
                                         .run = run,
                                         .reshape = NULL,
                                         .postrun = postrun,
                                         .init_node = init_node,
                                         .release_node = release_node,
                                         .score = score};

int register_add_n_hcl_x86_op()
{
    return register_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
}

int unregister_add_n_hcl_x86_op()
{
    return unregister_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
