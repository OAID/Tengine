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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <string>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

#define DIMENSION3

#ifdef DIMENSION2
#if 0 
float i_buf[]={
//0   1    2    3   4    5
1.1, 9.2, 3.3, 4.4, 5.5, 6.6,
3.1, 1.2, 1.21,6.1, 3.3, 3.2,
9.1, 4.2, 2.21,2.1, 4.3, 8.2,
3.1, 5.2, 5.21,3.1, 5.3, 6.2,
4.1, 7.2, 7.21,6.1, 0.3, 6.2,
7.1, 8.2, 3.21,0.1, 1.3, 3.2,
1.1, 4.2, 8.21,8.1, 2.3, 0.2};
#endif
#endif
#ifdef DIMENSION3
#if 0 
float i_buf[]={
1,2,3,
6,5,4,
10,11,12,
9,8,7
};
#endif
float i_buf[] = {
    0.27119219, 0.49740695, 0.7943446, 0.2445141,  0.3397804,  0.0236984, 0.3027298,  0.22311826, 0.5661515,

    0.47340485, 0.17199108, 0.6026132, 0.9333439,  0.91916169, 0.0674293, 0.02879724, 0.41228136, 0.8658179,

    0.02530401, 0.27181205, 0.6658000, 0.16099804, 0.32075416, 0.7399204, 0.66059157, 0.16099285, 0.9637474,

    0.0874474,  0.51338821, 0.7592801, 0.93077148, 0.33635355, 0.7292651, 0.47397384, 0.355015,   0.1230771};

#endif

int create_input_node(graph_t graph, const char* node_name, int h, int w, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

#ifdef DIMENSION2
    int dims[2] = {7, 6};
#endif
#ifdef DIMENSION3
    int dims[3] = {4, 3, 3};
#endif
    set_tensor_shape(tensor, dims, sizeof(dims) / sizeof(int));

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t test_node = create_graph_node(graph, node_name, "TopKV2");
    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    int data_type = get_tensor_data_type(input_tensor);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    int axis = 2;
    set_node_attr_int(test_node, "k", &axis);

    set_node_input_tensor(test_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor1 = create_graph_tensor(graph, "Value", data_type);
    tensor_t output_tensor2 = create_graph_tensor(graph, "Index", TENGINE_DT_INT32);

    set_node_output_tensor(test_node, 0, output_tensor1, TENSOR_TYPE_VAR);
    set_node_output_tensor(test_node, 1, output_tensor2, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor1);
    release_graph_tensor(output_tensor2);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int h, int w, int layout, int data_type)
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);

    if(graph == nullptr)
    {
        std::cerr << "create failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_layout(graph, layout) < 0)
    {
        std::cerr << "set layout failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }
    //   graphDumpGraph();
    const char* input_name = "data";

    if(create_input_node(graph, input_name, h, w, data_type) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_test_node(graph, test_node_name, input_name) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

void* set_input_data(graph_t graph)
{
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    //    void* i_buf = malloc(buf_size);

    int dims[4];

    get_tensor_shape(input_tensor, dims, 4);

#ifdef DIMENSION2
    int elem_num = dims[0] * dims[1];
#endif
#ifdef DIMENSION3
    int elem_num = dims[0] * dims[1] * dims[2];
#endif
    int data_type = get_tensor_data_type(input_tensor);

    for(int i = 0; i < elem_num; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf;
            f[i] = i_buf[i];
        }
        else if(data_type == TENGINE_DT_FP16)
        {
            __fp16* f16 = ( __fp16* )i_buf;

#ifdef __ARM_ARCH
            f16[i] = -1.0;
#else
            f16[i] = fp32_to_fp16(-1.1);
#endif
        }
        else if(data_type == TENGINE_DT_INT8)
        {
            int8_t* int8 = ( int8_t* )i_buf;
            int8[i] = -32;
        }
        else
        {
            uint8_t* i8 = ( uint8_t* )i_buf;
            i8[i] = 32;
        }
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);

    release_graph_tensor(input_tensor);

    return i_buf;
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    tensor_t output_tensor2 = get_node_output_tensor(test_node, 1);
    int dims[4];
    int elenum1 = 1;
    get_tensor_shape(output_tensor, dims, 4);
    for(int i = 0; i < 4; i++)
        printf("dims[%d]= %d    \n", i, dims[i]);
//    const TShape& in_shape = input_tensor->GetShape();
#ifdef DIMENSION2
    int dimension = 2;
#endif
#ifdef DIMENSION3
    int dimension = 3;
#endif
    for(int i = 0; i < dimension; i++)
    {
        if(dims[i] != 0)
            elenum1 *= dims[i];
    }
    printf("elenum1 = %d \n", elenum1);
#if 1
    void* o_buf = get_tensor_buffer(output_tensor);
    void* index_buf = get_tensor_buffer(output_tensor2);
    int data_type = get_tensor_data_type(output_tensor);
    for(int i = 0; i < elenum1; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* p = ( float* )o_buf;
            std::cout << i << " " << p[i] << "\n";
        }
        else if(data_type == TENGINE_DT_FP16)
        {
            __fp16* p = ( __fp16* )o_buf;

#ifdef __ARM_ARCH
            std::cout << i << " " << ( float )p[i] << "\n";
#else
            std::cout << i << " " << fp16_to_fp32(p[i]) << "\n";
#endif
        }
        else if(data_type == TENGINE_DT_INT8)
        {
            int8_t* p = ( int8_t* )o_buf;
            std::cout << i << " " << ( int )p[i] << "\n";
        }
        else
        {
            uint8_t* p = ( uint8_t* )o_buf;

            std::cout << i << " " << ( int )p[i] << "\n";
        }
    }
    std::cout << "Index buffer :\n";
    for(int j = 0; j < elenum1; j++)
    {
        int* p1 = ( int* )index_buf;
        std::cout << j << " " << p1[j] << "\n";
    }
#endif
    release_graph_tensor(output_tensor);
    release_graph_tensor(output_tensor2);
}

int main(int argc, char* argv[])
{
    int h = 7, w = 6;
    const char* test_node_name = "TopKV2";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NHWC;
    // float negative_slope=0.0;
    int res;
    while((res = getopt(argc, argv, "h:w:l:t:s")) != -1)
    {
        switch(res)
        {
            case 'h':
                h = strtoul(optarg, NULL, 10);
                break;

            case 'w':
                w = strtoul(optarg, NULL, 10);
                break;

            case 'l':
                layout = strtoul(optarg, NULL, 10);
                break;

            case 't':
                data_type = strtoul(optarg, NULL, 10);
                break;

            default:
                break;
        }
    }

    init_tengine();
    std::cout << "This is " << test_node_name << "Test\n";
    graph_t graph = create_test_graph(test_node_name, h, w, layout, data_type);

    if(graph == nullptr)
        return 1;

    /* set input */
    set_input_data(graph);

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }

    node_t test_node = get_graph_node(graph, test_node_name);

    const char* dev = get_node_device(test_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    dump_output_data(test_node);

    std::cout << " Test reshape ...\n";

    /* change the input shape */
#if 0
#endif
    release_graph_node(test_node);

    //   postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}
