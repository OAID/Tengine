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

    int dims[2] = {h, w};

    set_tensor_shape(tensor, dims, 2);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}
int create_test_node(graph_t graph, const char* node_name, const char* input_name, const char* input_name2)
{
    node_t test_node = create_graph_node(graph, node_name, "Maximum");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    int data_type = get_tensor_data_type(input_tensor);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);

    tensor_t input_tensor2 = get_graph_tensor(graph, input_name2);

    if(input_tensor2 == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 1, input_tensor2);

    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor2);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    // set_node_attr_float(test_node, "negative_slope", &negative_slope);
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

    const char* input_name = "data";

    if(create_input_node(graph, input_name, h, w, data_type) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    const char* input_name2 = "data2";

    if(create_input_node(graph, input_name2, h, w, data_type) < 0)
    {
        std::cerr << "create input2 failed\n";
        return nullptr;
    }

    if(create_test_node(graph, test_node_name, input_name, input_name2) < 0)
    {
        std::cerr << "create test node failed\n";
        return nullptr;
    }

    /* set input/output node */
    const char* inputs[] = {input_name, input_name2};
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
    void* i_buf = malloc(buf_size);
    void* i_buf1 = malloc(buf_size);

    int dims[4];

    get_tensor_shape(input_tensor, dims, 4);

    int elem_num = dims[0] * dims[1];

    std::cout << "Dims[0]=" << dims[0] << " dims[1]=" << dims[1] << "elem_num=" << elem_num << "\n";
    tensor_t input_tensor1 = get_graph_input_tensor(graph, 1, 0);

    int data_type = get_tensor_data_type(input_tensor);
    std::cout << "data type = " << data_type;
    for(int i = 0; i < elem_num; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf;
            f[i] = 0.3907;
            float* p = ( float* )i_buf1;
            p[i] = 0.98765;
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
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);

    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor1);

    return i_buf;
}

void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);

    for(int i = 0; i < dims[0] * dims[1]; i++)
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

    release_graph_tensor(output_tensor);
}

int main(int argc, char* argv[])
{
    int h = 6, w = 7;
    const char* test_node_name = "Maximum";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;
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

    std::cout << "hubin on running on Mamimum\n";
    init_tengine();

    graph_t graph = create_test_graph(test_node_name, h, w, layout, data_type);

    if(graph == nullptr)
        return 1;

    /* set input */
    //    void* i_buf = set_input_data(graph);
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
    int dims[2] = {h * 2, w * 2};

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    set_tensor_shape(input_tensor, dims, 2);

    free(i_buf);

    i_buf = set_input_data(graph);

    if(run_graph(graph, 1) < 0)
    {
        std::cerr << "run_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 2;
    }

    dump_output_data(test_node);
#endif
    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();
    return 0;
}
