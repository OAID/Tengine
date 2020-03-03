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
        if(off!=0)
        {
            // std::cout <<"mismatch:\t["<<i<<"]\ta:"<<a[i] <<"\tb:"<<b[i]<<"\toff:"<<a[i]-b[i]<<"\n";
            break;
        }
    }
    if(i!= size)
    {
        printf("mismatch:\n\t[%d]\t---a:    %f ,%f   :b---        off: %f\n",i,a[i],b[i],a[i]-b[i]);
        return -1;
    }
    return 0;
}

int fp16_mismatch(__fp16* a, __fp16* b, int size)
{
    int i =0;

    for(i=0;i<size;i++)
    {
        float off = fp16_to_fp32(a[i]) - fp16_to_fp32(b[i]);
        if(off!=0)
        {
            // std::cout <<"mismatch:\t["<<i<<"]\ta:"<<a[i] <<"\tb:"<<b[i]<<"\toff:"<<a[i]-b[i]<<"\n";
            break;
        }
    }
    if(i!= size)
    {
        printf("mismatch:\n\t[%d]\t---a:    %f ,%f   :b---        off: %f\n",i,fp16_to_fp32(a[i]),fp16_to_fp32(b[i]),fp16_to_fp32(a[i]) - fp16_to_fp32(b[i]));
        return -1;
    }
    return 0;
}

int create_input_node(graph_t graph, const char* node_name, int c, int h, int w, int type_from)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    int data_type = 0;
    if(type_from == 1) 
        data_type = TENGINE_DT_FP32;
    else if(type_from == 2) 
        data_type = TENGINE_DT_FP16;
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
int create_test_node(graph_t graph, const char* node_name, const char* input_name, int type_from, int type_to)
{
    node_t test_node = create_graph_node(graph, node_name, "Cast");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(test_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    int data_type = 0;
    if(type_to == 1) 
        data_type = TENGINE_DT_FP32;
    else if(type_to == 2) 
        data_type = TENGINE_DT_FP16;
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    set_node_attr_int(test_node, "type_from", &type_from);
    set_node_attr_int(test_node, "type_to", &type_to);
    release_graph_node(test_node);

    return 0;
}

graph_t create_test_graph(const char* test_node_name, int c, int h, int w, int layout, int type_from, int type_to)
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

    if(create_input_node(graph, input_name, c, h, w, type_from) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_test_node(graph, test_node_name, input_name, type_from, type_to) < 0)
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



void dump_output_data(node_t test_node, int type_to)
{
    printf("run dump\n");
    tensor_t output_tensor = get_node_output_tensor(test_node, 0);
    
     int dims[4];

    get_tensor_shape(output_tensor, dims, 4);
    if(type_to == 1)
    {
        float* o_buf = (float*)get_tensor_buffer(output_tensor);
        for(int i = 0; i < dims[0]*dims[1]*dims[2]*dims[3]; i++)
        {
            std::cout << " " << o_buf[i];
            if((i+1)%32==0)printf("\n");
        }
    }
    else if(type_to == 2)
    {
        __fp16* o_buf = (__fp16*)get_tensor_buffer(output_tensor);
        for(int i = 0; i < dims[0]*dims[1]*dims[2]*dims[3]; i++)
        {
            std::cout << " " << fp16_to_fp32(o_buf[i]);
            if((i+1)%32==0)printf("\n");
        }
    }

    release_graph_tensor(output_tensor);
}

int test_cast(int c, int h, int w, const char* test_node_name, int layout, int type_from, int type_to)
{
    graph_t graph = create_test_graph(test_node_name, c, h, w, layout, type_from, type_to);

    if(graph == nullptr)
        return -1;

    /* set input */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int data_type = get_tensor_data_type(input_tensor);

    int buf_size = get_tensor_buffer_size(input_tensor);

    void* i_buf = malloc(buf_size);
    if(data_type == TENGINE_DT_FP32)
    {
        float* f = ( float* )i_buf;
        FILE *infp;  
        infp=fopen("./data/cast_fp32input.bin","rb");
        if(fread(f, sizeof(float), buf_size/sizeof(float), infp) == 0)
        {
            printf("read input failed\n");
            return false;
        }
        fclose(infp);
        // for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
        // {
        //     f[i] = i;
        // }
        // FILE *outfp;  
        // outfp=fopen("/home/dongdong/fp32input.bin","wb");
        // fwrite(f,sizeof(float),buf_size/sizeof(float),outfp);
        // fclose(outfp);

    }
    else if(data_type == TENGINE_DT_FP16)
    {
        __fp16* f16 = ( __fp16* )i_buf;
        FILE *infp;  
        infp=fopen("./data/cast_fp16input.bin","rb");
        if(fread(f16, sizeof(__fp16), buf_size/sizeof(__fp16), infp) == 0)
        {
            printf("read input failed\n");
            return false;
        }
        fclose(infp);
        // for(unsigned int i = 0; i < buf_size/sizeof(__fp16) ; i++)
        // {
        //     f16[i] = fp32_to_fp16(i);
        // }

        // FILE *outfp;  
        // outfp=fopen("/home/dongdong/fp16input.bin","wb");
        // fwrite(f16,sizeof(__fp16),buf_size/sizeof(__fp16),outfp);
        // fclose(outfp);
    }
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
    data_type = get_tensor_data_type(output_tensor);

    if(type_to == 1)
    {
        float* buf = (float*)get_tensor_buffer(output_tensor);
        float* buf1 = ( float* )malloc(size);
        FILE *infp;  
        infp=fopen("./data/cast_fp32output.bin","rb");
        if(fread(buf1, sizeof(float), size/sizeof(float), infp)==0)
        {
            printf("read input failed\n");
            return false;
        }
        fclose(infp);

        if(float_mismatch(buf, buf1, size/sizeof(float)) != 0)
            return -1;

    }
    else if(type_to == 2)
    {
        __fp16* buf = (__fp16*)get_tensor_buffer(output_tensor);
        __fp16* buf1 = ( __fp16* )malloc(size);
        FILE *infp;  
        infp=fopen("./data/cast_fp16output.bin","rb");
        if(fread(buf1, sizeof(__fp16), size/sizeof(__fp16), infp)==0)
        {
            printf("read input failed\n");
            return false;
        }
        fclose(infp);

        if(fp16_mismatch(buf, buf1, size/sizeof(float)) != 0)
            return -1;

    }     
    // dump_output_data(test_node, type_to);
    free(i_buf);

    release_graph_node(test_node);

    postrun_graph(graph);
    destroy_graph(graph);
    return 0;
}

int main(int argc, char* argv[])
{
    int c = 4, h = 200, w = 200;
    const char* test_node_name = "cast";
    int layout = TENGINE_LAYOUT_NCHW;

    init_tengine();
    
    if(test_cast(c, h, w, test_node_name, layout, 2, 1) < 0)
    {
        printf("fail\n");
        return false;
    }
    if(test_cast(c, h, w, test_node_name, layout, 1, 2) < 0)
    {
        printf("fail\n");
        return false;
    }
        
    printf("pass\n");

    release_tengine();
    return 0;
}
