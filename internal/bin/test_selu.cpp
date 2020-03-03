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
 * Author: ddzhao@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <string>

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
        if(off!=0)
        {
            // std::cout <<"mismatch:\t["<<i<<"]\ta:"<<a[i] <<"\tb:"<<b[i]<<"\toff:"<<a[i]-b[i]<<"\n";
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
int create_test_node(graph_t graph, const char* node_name, const char* input_name, int data_type, float alpha, float lambda)
{
    node_t test_node = create_graph_node(graph, node_name, "Selu");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

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

    set_node_attr_float(test_node, "alpha", &alpha);
    set_node_attr_float(test_node, "lambda", &lambda);
    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int c, int h, int w, int layout, int data_type, float alpha, float lambda)
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

    if(create_test_node(graph, test_node_name, input_name, data_type, alpha, lambda) < 0)
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



void dump_output_data(node_t test_node)
{
    printf("run dump\n");
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    
     int dims[4];

    get_tensor_shape(output_tensor, dims, 4);

    float* o_buf = (float*)get_tensor_buffer(output_tensor);
    for(int i = 0; i < dims[0]*dims[1]*dims[2]*dims[3]; i++)
    {
        std::cout << " " << o_buf[i];
        if((i+1)%32==0)printf("\n");
    }
    

    release_graph_tensor(output_tensor);
}

int test_op(int c, int h, int w, const char* test_node_name, int layout, int data_type, float alpha, float lambda)
{
    graph_t graph = create_test_graph(test_node_name, c, h, w, layout, data_type, alpha, lambda);

    if(graph == nullptr)
        return -1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int buf_size = get_tensor_buffer_size(input_tensor);
    float* i_buf = ( float* )malloc(buf_size);

    FILE *infp;  
    infp=fopen("./data/selu_in.bin","rb");
    if(fread(i_buf, sizeof(float), buf_size/sizeof(float), infp)==0)
    {
        printf("read input data file failed!\n");
        return false;
    }
    fclose(infp);

    // for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
    // {
    //     i_buf[i] = i*0.1-5;
    // }
    set_tensor_buffer(input_tensor, i_buf, buf_size);
    release_graph_tensor(input_tensor);

    if(prerun_graph(graph) < 0)
    {
        std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
        return 1;
    }
    // get device
    node_t test_node = get_graph_node(graph, test_node_name);
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

    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    int size = get_tensor_buffer_size(output_tensor);
    float* buf = ( float* )get_tensor_buffer(output_tensor);
    float* buf1 = ( float* )malloc(size);

    infp=fopen("./data/selu_out.bin","rb");
    if(fread(buf1, sizeof(float), size/sizeof(float), infp)==0)
    {
        printf("read output data file failed!\n");
        return false;
    }
    fclose(infp);
    if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
        return -1;

    // dump_output_data(test_node);
    free(i_buf);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);
    return 0;
}

int main(int argc, char* argv[])
{
    int c = 1, h = 10, w = 10;
    const char* test_node_name = "selu";
    int data_type = TENGINE_DT_FP32;
    int layout = TENGINE_LAYOUT_NCHW;
    float alpha = 1.67326324f;
    float lambda = 1.050700987f;

    init_tengine();
    
    test_op(c, h, w, test_node_name, layout, data_type, alpha, lambda);

    

    release_tengine();
    return 0;
}