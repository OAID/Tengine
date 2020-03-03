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
 * Author: haoluo@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <string.h>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

#include <unistd.h>
#include <iostream>
#include <string>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

int create_input_node(graph_t graph)
{
    node_t node = create_graph_node(graph, "Input", "InputOp");
    tensor_t tensor = create_graph_tensor(graph, "Input", TENGINE_DT_FP32);

    // printf("create_input_node [%d]:[%d]:[%d]:[%d]\n",n,c,h,w);
    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    // int dims[3] = { 2, 2, 2};
    int dims[2] = {1, 3};

    set_tensor_shape(tensor, dims, 2);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph)
{
    node_t test_node = create_graph_node(graph, "swap", "SwapAxis");
    if(test_node == nullptr)
    {
        std::cout << "create_graph_node_failed : node_name\n";
        return -1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, "Input");
    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\noutput";
        return -1;
    }

    // std::cout << "set_node_input_tensor\n";
    set_node_input_tensor(test_node, 0, input_tensor);
    // std::cout << "release_input_tensor\n";
    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, "swap", TENGINE_DT_FP32);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    int dim0 = 0;
    int dim1 = 2;
    set_node_attr_int(test_node, "dim_0", &dim0);
    set_node_attr_int(test_node, "dim_1", &dim1);
    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph()
{
    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    if(graph == nullptr)
    {
        std::cerr << "create failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(create_input_node(graph) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    // std::cout << "create_test_node\n";
    create_test_node(graph);

    const char* outputs[] = {"swap"};
    const char* inputs[] = {"Input"};
    // std::cout << "set_graph_input_node\n";
    if(set_graph_input_node(graph, inputs, 1) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    // std::cout << "set_graph_output_node\n";
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
    // float buf[8] = {0,1,2,3,4,5,6,7};
    float buf[3] = {1, 2, 3};
    void* i_buf = malloc(buf_size);

    int dims[4];

    int num = get_tensor_shape(input_tensor, dims, 4);
    printf("input: buf_size: %d ---> shape: ", buf_size);
    for(int i = 0; i < num; i++)
    {
        printf("%d,", dims[i]);
    }
    printf("\n");
    memcpy(i_buf, buf, buf_size);
    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    return i_buf;
}

int test_swap(void)
{
    graph_t graph = create_test_graph();
    if(graph == nullptr)
    {
        std::cout << "create graph failed!!!\n";
        return 1;
    }
    int ret = prerun_graph(graph);
    printf("prerun: %d\n", ret);
    void* i_buf = set_input_data(graph);

    ret = run_graph(graph, 1);
    printf("run: %d\n", ret);

    tensor_t o_tensor = get_graph_output_tensor(graph, 0, 0);

    int dims[4];

    int num = get_tensor_shape(o_tensor, dims, 4);
    printf("output =========  shape[%d]: ", num);
    for(int i = 0; i < num; i++)
    {
        printf("%d,", dims[i]);
    }
    printf("\n buf:");
    float* o_buf = ( float* )get_tensor_buffer(o_tensor);

    int size = get_tensor_buffer_size(o_tensor) / 4;
    for(int i = 0; i < size; i++)
        printf("%f,", o_buf[i]);
    printf("\n");

    ret = postrun_graph(graph);
    free(i_buf);
    destroy_graph(graph);

    return ret;
}

int main(int argc, char* argv[])
{
    std::cout << "init_tengine\n";
    init_tengine();

    test_swap();

    release_tengine();

    return 0;
}
