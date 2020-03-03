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

#define DIMENSION2

#ifdef DIMENSION2
float i_buf[] = {1.1,  9.2, 3.3, 4.4, 5.5,  6.6, 3.1,  1.2, 1.21, 6.1, 3.3,  3.2, 9.1,  4.2,
                 2.21, 2.1, 4.3, 8.2, 3.1,  5.2, 5.21, 3.1, 5.3,  6.2, 4.1,  7.2, 7.21, 6.1,
                 0.3,  6.2, 7.1, 8.2, 3.21, 0.1, 1.3,  3.2, 1.1,  4.2, 8.21, 8.1, 2.3,  0.2};

#endif
#ifdef DIMENSION3
float i_buf[] = {1, 2, 3, 6, 5, 4, 10, 11, 12, 9, 8, 7};

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
    int dims[3] = {2, 2, 3};
#endif
    set_tensor_shape(tensor, dims, sizeof(dims) / sizeof(int));

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}
int create_test_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t test_node = create_graph_node(graph, node_name, "ArgMin");
    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    int data_type = get_tensor_data_type(input_tensor);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    int axis = 1;
    std::cout << "Axis -" << axis << "\n";
    set_node_attr_int(test_node, "axis", &axis);
#ifdef DIMENSION3
    int dimensions = 3;
#endif
#ifdef DIMENSION2
    int dimensions = 2;
#endif
    set_node_attr_int(test_node, "dimension", &dimensions);

    set_node_input_tensor(test_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

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
    for(int i = 0; i < dimension - 1; i++)
    {
        if(dims[i] != 0)
            elenum1 *= dims[i];
    }
#if 1
    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);
    for(int i = 0; i < elenum1; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            int* p = ( int* )o_buf;
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
#endif
    release_graph_tensor(output_tensor);
}

int main(int argc, char* argv[])
{
    int h = 7, w = 6;
    const char* test_node_name = "ArgMin";
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
    std::cout << " Test reshape ..." << __LINE__ << "\n";
    release_graph_node(test_node);
    std::cout << " Test reshape ..." << __LINE__ << "\n";

    //   postrun_graph(graph);
    std::cout << " Test reshape ..." << __LINE__ << "\n";
    destroy_graph(graph);
    std::cout << " Test reshape ..." << __LINE__ << "\n";

    release_tengine();
    std::cout << " Test reshape ..." << __LINE__ << "\n";
    return 0;
}
