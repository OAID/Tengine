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
 * Copyright (c) 2019, Open AI Lab
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>           

#include "tengine/c_api.h"
#include "tengine_c_api_ex.h"

int allocated_num = 0;
void** record_ptr = NULL;

void record_allocated_buf(void* buf)
{
    allocated_num++;
    record_ptr = realloc(record_ptr, sizeof(void*) * allocated_num);
    record_ptr[allocated_num - 1] = buf;
}

void free_allocated_buf(void)
{
    for(int i = 0; i < allocated_num; i++)
        free(record_ptr[i]);

    if(record_ptr)
        free(record_ptr);
}

void init_buffer(void* buf, int elem_num, int elem_size, int val)
{
    for(int i = 0; i < elem_num; i++)
    {
        float val0;
        float* fp;
        int16_t* i16;
        char* c;

        if(val >= 0)
            val0 = val;
        else
            val0 = i%10;

        switch(elem_size)
        {
            case 4:
                fp = ( float* )buf;
                fp[i] = val0;
                break;
            case 2:
                i16 = ( int16_t* )buf;
                i16[i] = val0;
                break;
            case 1:
                c = ( char* )buf;
                c[i] = val0;
                break;
        }
    }
}

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int k_size, int stride, int pad,
                     int in_c, int out_c, int group)
{
    /* weight */
    char* weight_name = malloc(strlen(node_name) + 16);
    sprintf(weight_name, "%s/weight", node_name);

    node_t w_node = create_graph_node(graph, weight_name, "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name, TENGINE_DT_FP32);

    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    int w_dims[] = {out_c, in_c / group, k_size, k_size};
    set_tensor_shape(w_tensor, w_dims, 4);

    /* bias */

    char* bias_name = malloc(strlen(node_name) + 16);
    sprintf(bias_name, "%s/bias", node_name);

    node_t b_node = create_graph_node(graph, bias_name, "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name, TENGINE_DT_FP32);

    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    /* conv */

    node_t conv_node = create_graph_node(graph, node_name, "Convolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == NULL)
    {
        fprintf(stderr, "errno= %d\n", get_tengine_errno());
        return -1;
    }

    set_node_input_tensor(conv_node, 2, b_tensor);
    set_node_input_tensor(conv_node, 1, w_tensor);
    set_node_input_tensor(conv_node, 0, input_tensor);

    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    free(bias_name);
    free(weight_name);

    /* attr */
    set_node_attr_int(conv_node, "kernel_h", &k_size);
    set_node_attr_int(conv_node, "kernel_w", &k_size);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad);
    set_node_attr_int(conv_node, "pad_h1", &pad);
    set_node_attr_int(conv_node, "pad_w0", &pad);
    set_node_attr_int(conv_node, "pad_w1", &pad);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "input_channel", &in_c);
    set_node_attr_int(conv_node, "group", &group);

    release_graph_node(conv_node);

    return 0;
}

int create_pooling_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t pool_node = create_graph_node(graph, node_name, "Pooling");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == NULL)
    {
        fprintf(stderr, "ERRNO: %d\n", get_tengine_errno());
        return -1;
    }

    set_node_input_tensor(pool_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(pool_node, 0, output_tensor, TENSOR_TYPE_VAR);
    release_graph_tensor(output_tensor);

    release_graph_node(pool_node);

    return 0;
}

graph_t create_test_graph(int c, int h, int w, int out_c)
{
    graph_t graph = create_graph(NULL, NULL, NULL);

    if(graph == NULL)
    {
        fprintf(stderr, "ERRNO: %d\n", get_tengine_errno());
        return NULL;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        fprintf(stderr, "create input failed\n");
        return NULL;
    }

    // int out_c = 4;
    //                                                k  s  p in_c out_c group
    if(create_conv_node(graph, conv_name, input_name, 1, 1, 0, c, out_c, 1) < 0)
    {
        fprintf(stderr, "create conv node failed\n");
        return NULL;
    }
#if 0
    const char* pool_name = "pooling";

    if(create_pooling_node(graph, pool_name, conv_name) < 0)
    {
        fprintf(stderr, "create pooling node failed\n");
        return NULL;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {pool_name};
#else
    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};

#endif

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed: ERRNO: %d\n", get_tengine_errno());
        return NULL;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed: ERRNO: %d\n", get_tengine_errno());
        return NULL;
    }

    return graph;
}

void fill_conv_node(node_t node)
{
    tensor_t filter = get_node_input_tensor(node, 1);
    int dims[4];

    get_tensor_shape(filter, dims, 4);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int elem_size = 4;

    void* filter_buf = malloc(elem_num * elem_size);

    init_buffer(filter_buf, elem_num, elem_size, -1);

    set_tensor_buffer(filter, filter_buf, elem_num * elem_size);

    record_allocated_buf(filter_buf);

    release_graph_tensor(filter);

    tensor_t bias = get_node_input_tensor(node, 2);

    if(bias == NULL)
        return;

    get_tensor_shape(bias, dims, 1);

    elem_num = dims[0];

    void* bias_buf = malloc(elem_num * elem_size);

    init_buffer(bias_buf, elem_num, elem_size, 3);

    set_tensor_buffer(bias, bias_buf, elem_num * elem_size);

    record_allocated_buf(bias_buf);

    release_graph_tensor(bias);
}

void fill_graph_param(graph_t graph)
{
    int node_num = get_graph_node_num(graph);

    for(int i = 0; i < node_num; i++)
    {
        node_t node = get_graph_node_by_idx(graph, i);

        const char* node_op = get_node_op(node);

        if(!strcmp(node_op, "Convolution"))
        {
            fill_conv_node(node);
        }

        release_graph_node(node);
    }
}

int main(int argc, char* argv[])
{
    int c, h, w, out_c;

    c = 8;
    h = 14;
    w = 14;
    out_c = 16;

    init_tengine();

    graph_t graph = create_test_graph(c, h, w, out_c);
 
    if(graph == NULL)
        return 1;

    fill_graph_param(graph);

    /* fill input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dims[4];
    int dim_num = get_tensor_shape(input_tensor, dims, 4);

    int elem_num = 1;
    int elem_size = 4;

    for(int i = 0; i < dim_num; i++)
        elem_num *= dims[i];

    void* input_buf = malloc(elem_num * elem_size);

    init_buffer(input_buf, elem_num, elem_size, -1);
    record_allocated_buf(input_buf);

    set_tensor_buffer(input_tensor, input_buf, elem_num * elem_size);
    release_graph_tensor(input_tensor);

    prerun_graph(graph);

    dump_graph(graph);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    dim_num = get_tensor_shape(output_tensor, dims, 4);

    elem_num = 1;

    printf("output shape: [");

    for(int i = 0; i < dim_num; i++)
    {
        elem_num *= dims[i];
        printf(" %d", dims[i]);
    }

    printf(" ]\n");

    float* output = get_tensor_buffer(output_tensor);

    for(int i = 0; i < elem_num; i++)
    {
        int w = dims[3];

        if((i % w) == 0)
            printf("\n%d:\t", i);

        printf(" %f", output[i]);
    }

    printf("\n");

    release_graph_tensor(output_tensor);

    postrun_graph(graph);

    destroy_graph(graph);

    release_tengine();

    free_allocated_buf();

    return 0;
}
