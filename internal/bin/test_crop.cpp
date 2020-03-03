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
 * Author: zpluo@openailab.com
           bingzhang@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <string>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node_1(graph_t graph, const char* node_name, int c, int h, int w, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, h, w, c};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node_Case1(graph_t graph, const char* node_name, int c, int h, int w, int data_type)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);

    if(tensor == nullptr)
    {
        release_graph_node(node);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h - 1, w - 1};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_test_node(graph_t graph, const char* node_name, const char* input_name, const char* input_name_1, int i_type)
{
    node_t test_node = create_graph_node(graph, node_name, "Crop");

    const int crop_h = 1.0f;
    const int crop_w = 0.0f;
    set_node_attr_int(test_node, "crop_h", &crop_h);
    set_node_attr_int(test_node, "crop_w", &crop_w);
    const int offset_h = 1.0f;
    const int offset_w = 1.0f;
    set_node_attr_int(test_node, "offset_h", &offset_h);
    set_node_attr_int(test_node, "offset_w", &offset_w);
    const int num_args = 2;
    set_node_attr_int(test_node, "num_args", &num_args);

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    tensor_t input_tensor_1 = get_graph_tensor(graph, input_name_1);
    int data_type = get_tensor_data_type(input_tensor);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);
    set_node_input_tensor(test_node, 1, input_tensor_1);

    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor_1);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int c, int h, int w, int layout, int data_type, int i_type,
                          int case_num)
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

    const char* input_name = "data";
    const char* input_name_1 = "data1";
    if(layout == 0)
    {
        if(create_input_node(graph, input_name, c, h, w, data_type) < 0)
        {
            std::cerr << "create input failed\n";
            return nullptr;
        }
    }
    else
    {
        if(create_input_node_1(graph, input_name, c, h, w, data_type) < 0)
        {
            std::cerr << "create input failed\n";
            return nullptr;
        }
    }

    if(create_input_node_Case1(graph, input_name_1, c, h, w, data_type) < 0)
    {
        std::cerr << "create input1 failed\n";
        return nullptr;
    }

    if(create_test_node(graph, test_node_name, input_name, input_name_1, i_type) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name, input_name_1};
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

    // input_tensor_1

    void* i_buf = malloc(buf_size);

    int dims[4];

    get_tensor_shape(input_tensor, dims, 4);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int data_type = get_tensor_data_type(input_tensor);

    for(int i = 0; i < elem_num; i++)
    {
        float* f = ( float* )i_buf;
        f[i] = i;
    }
    set_tensor_buffer(input_tensor, i_buf, buf_size);

    release_graph_tensor(input_tensor);

    if(data_type == TENGINE_DT_UINT8 || data_type == TENGINE_DT_INT8)
    {
        float scale = 0.1;
        int zero = 30;
        set_tensor_quant_param(input_tensor, &scale, &zero, 1);
    }
    return i_buf;
}

void* set_input_data_case1(graph_t graph)
{
    tensor_t input_tensor_1 = get_graph_input_tensor(graph, 1, 0);
    // tensor_t input_tensor_1=get_node_input_tensor();

    int buf_size_1 = get_tensor_buffer_size(input_tensor_1);

    // input_tensor_1
    void* i_buf_1 = malloc(buf_size_1);

    int dims[4];

    get_tensor_shape(input_tensor_1, dims, 4);
    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int data_type = get_tensor_data_type(input_tensor_1);

    for(int i = 0; i < elem_num; i++)
    {
        float* f = ( float* )i_buf_1;
        f[i] = i;
    }
    set_tensor_buffer(input_tensor_1, i_buf_1, buf_size_1);

    release_graph_tensor(input_tensor_1);

    if(data_type == TENGINE_DT_UINT8 || data_type == TENGINE_DT_INT8)
    {
        float scale = 0.1;
        int zero = 30;
        set_tensor_quant_param(input_tensor_1, &scale, &zero, 1);
    }

    return i_buf_1;
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);
    float scale = 0.1f;
    int zero = 0;
    if(data_type == TENGINE_DT_INT8 || data_type == TENGINE_DT_UINT8)
    {
        get_tensor_quant_param(output_tensor, &scale, &zero, 1);
        printf("output  scale: %f ,zero: %d\n", scale, zero);
    }

    for(int i = 0; i < dims[1] * dims[0] * dims[2] * dims[3]; i++)
    {
        float* p = ( float* )o_buf;
        std::cout << i << " " << p[i] << "\n";
    }

    release_graph_tensor(output_tensor);
}

int main(int argc, char* argv[])
{
    int c = 2, h = 5, w = 5;
    const char* test_node_name = "test";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;
    int i_type = 2;
    int case_nu = 1;
    int res;

    while((res = getopt(argc, argv, "h:w:l:t:e:c:")) != -1)
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

            case 'e':
                i_type = strtoul(optarg, NULL, 10);
                break;

            case 'c':
                case_nu = strtoul(optarg, NULL, 10);
                break;

            default:
                break;
        }
    }

    init_tengine();

    graph_t graph = create_test_graph(test_node_name, c, h, w, layout, data_type, i_type, case_nu);

    if(graph == nullptr)
        return 1;

    /* set input */
    void* i_buf = set_input_data(graph);
    void* i_buf_1 = nullptr;

    i_buf_1 = set_input_data_case1(graph);

    tensor_t output = get_graph_output_tensor(graph, 0, 0);
    if(data_type == 3)
    {
        float scale = 0.1;
        int zero = 0;
        set_tensor_quant_param(output, &scale, &zero, 1);
    }

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    // dump_graph(graph);
    node_t test_node = get_graph_node(graph, test_node_name);

    const char* dev = get_node_device(test_node);

    std::cout << "node running on dev: " << dev << "\n";

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }
    // std::cout<<"Pass test\n";
    dump_output_data(test_node);
    free(i_buf);
    free(i_buf_1);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}
