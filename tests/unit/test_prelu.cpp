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
#include <cmath>

#include "tengine_c_api.h"
#include "compiler_fp16.h"

static inline unsigned long get_cur_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

int float_mismatch(float* a, float* b, int size)
{
    int i =0;
    for(i=0;i<size;i++)
    {
        float off = a[i] - b[i];
        if(std::abs(off) > 1e-4)
        {
            // printf("mismatch:\n\t[%d]\t---a:    %f ,%f   :b---        off: %f\n",i,a[i],b[i],a[i]-b[i]);
            break;
        }
    }
    if(i!= size)
    {
        printf("mismatch:\n\t[%d]\t---a:    %f ,%f   :b---        off: %f\n",i,a[i],b[i],a[i]-b[i]);
        printf("fail\n");
        return -1;
    }
    printf("pass\n");
    return 0;
}

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
int create_test_node(graph_t graph, const char* node_name, const char* input_name)
{
    node_t test_node = create_graph_node(graph, node_name, "PReLU");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    int data_type = get_tensor_data_type(input_tensor);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    // std::string slope_name(node_name);
    // slope_name += "/slope";

    node_t s_node = create_graph_node(graph, "slope", "Const");
    tensor_t s_tensor = create_graph_tensor(graph, "slope", TENGINE_DT_FP32);
    set_node_output_tensor(s_node, 0, s_tensor, TENSOR_TYPE_CONST);

    set_node_input_tensor(test_node, 1, s_tensor);
    int dims[4];
    get_tensor_shape(input_tensor, dims, 4);
    int dim1 = dims[1];
    int s_dims[1] = {dim1};

    set_tensor_shape(s_tensor, s_dims, 1);

    release_graph_node(s_node);
    release_graph_tensor(s_tensor);

    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int c, int h, int w, int layout, int data_type)
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

    if(create_input_node(graph, input_name, c, h, w, data_type) < 0)
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

    // input_tensor

    void* i_buf = malloc(buf_size);

    int dims[4];

    get_tensor_shape(input_tensor, dims, 4);

    int elem_num = dims[0] * dims[1] * dims[2] * dims[3];
    int data_type = get_tensor_data_type(input_tensor);

    for(int i = 0; i < elem_num; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* f = ( float* )i_buf;
            f[i] = -11;
        }
        else if(data_type == TENGINE_DT_INT8)
        {
            int8_t* int8 = ( int8_t* )i_buf;
            int8[i] = -11;
        }
        else
        {
            uint8_t* i8 = ( uint8_t* )i_buf;
            i8[i] = 20;
        }
    }
    if(data_type == TENGINE_DT_UINT8 || data_type == TENGINE_DT_INT8)
    {
        float scale = 0.1;
        int zero = 30;
        set_tensor_quant_param(input_tensor, &scale, &zero, 1);
    }

    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    return i_buf;
}


void dump_output_data(node_t test_node)
{
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    void* o_buf = get_tensor_buffer(output_tensor);
    int data_type = get_tensor_data_type(output_tensor);
    printf("datatype:%d\n", data_type);

    for(int i = 0; i < dims[1] * dims[0] * dims[2] * dims[3]; i++)
    {
        if(data_type == TENGINE_DT_FP32)
        {
            float* p = ( float* )o_buf;
            std::cout <<" " << p[i];
            if(i%32==0)printf("\n");
        }

    }

    release_graph_tensor(output_tensor);
}

int main(int argc, char* argv[])
{
    int c = 64, h = 224, w = 224;
    const char* test_node_name = "PReLU";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;

    init_tengine();

    graph_t graph = create_test_graph(test_node_name, c, h, w, layout, data_type);
    graph_t graph1 = create_test_graph(test_node_name, c, h, w, layout, data_type);

    if(graph == nullptr || graph1 == nullptr)
        return -1;

    /* set input *////////////////////////////////////////////////////////////
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph1, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);
    float* i_buf1 = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
    {
        i_buf[i] = rand();
        i_buf1[i] = i_buf[i];
    }


    set_tensor_buffer(input_tensor, i_buf, buf_size);
    set_tensor_buffer(input_tensor1, i_buf1, buf_size);
    release_graph_tensor(input_tensor);
    release_graph_tensor(input_tensor1);

    // set slope///////////////////////////////////////////////////////////////

    node_t test_node = get_graph_node(graph, test_node_name);
    node_t test_node1 = get_graph_node(graph1, test_node_name);

    tensor_t slope_tensor = get_node_input_tensor(test_node, 1);
    tensor_t slope_tensor1 = get_node_input_tensor(test_node1, 1);

    buf_size = get_tensor_buffer_size(slope_tensor);
    float* s_buf = ( float* )malloc(buf_size);
    float* s_buf1 = ( float* )malloc(buf_size);

    for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
    {
        s_buf[i] = 0.1;
        s_buf1[i] = 0.1;
    }

    set_tensor_buffer(slope_tensor, s_buf, buf_size);
    set_tensor_buffer(slope_tensor1, s_buf1, buf_size);
    release_graph_tensor(slope_tensor);
    release_graph_tensor(slope_tensor1);

    // prerun graph
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    // prerun graph1
    setenv("OPS_REGISTRY","reference",1);
    setenv("OP_NAME","PReLU",1);
    if(prerun_graph(graph1) < 0 )
    {
        std::cerr << "prerun_graph1 failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    unsetenv("OPS_REGISTRY");
    unsetenv("OP_NAME");

    // get device
    // const char* dev = get_node_device(test_node);

    // std::cout << "node running on dev: " << dev << "\n";

    // run graph and time calc
    int repeat_count = 1;
    const char * rep_str=std::getenv("REPEAT");
    if(rep_str)
    repeat_count=strtoul(rep_str,NULL,10);

    // unsigned long start_time = get_cur_time();
    for(int i=0;i<repeat_count;i++)
    {
        run_graph(graph,1);
    }
    // unsigned long end_time = get_cur_time();
    // unsigned long off_time = end_time - start_time;
    // std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
    //             off_time);

    // start_time = get_cur_time();
    for(int i=0;i<repeat_count;i++)
    {
        run_graph(graph1,1000);
    }
    // end_time = get_cur_time();
    // off_time = end_time - start_time;
    // std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
    //             off_time);

    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    tensor_t output_tensor1 = get_node_output_tensor(test_node1, 0);
    int size = get_tensor_buffer_size(output_tensor);
    float* buf = ( float* )get_tensor_buffer(output_tensor);
    float* buf1 = ( float* )get_tensor_buffer(output_tensor1);
    if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
        return -1;

    // dump_output_data(test_node);
    // dump_output_data(test_node1);
    free(i_buf);
    free(s_buf);
    free(i_buf1);
    free(s_buf1);

    release_graph_node(test_node);
    release_graph_node(test_node1);

    postrun_graph(graph);
    destroy_graph(graph);
    postrun_graph(graph1);
    destroy_graph(graph1);

    release_tengine();
    return 0;
}
