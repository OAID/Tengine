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
 * Author: jjzeng@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <algorithm>

#include "tengine_c_api.h"

namespace AuthHelp
{
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


    int create_pool_node(graph_t graph, const char* node_name, const char* input_name, int kernel_h, int kernel_w,
                     int stride_h, int stride_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int method)
    {
        node_t pool_node = create_graph_node(graph, node_name, "Pooling");

        tensor_t input_tensor = get_graph_tensor(graph, input_name);

        if(input_tensor == nullptr)
        {
            std::cout << "ERRNO: " << get_tengine_errno() << "\n";
            return -1;
        }

        set_node_input_tensor(pool_node, 0, input_tensor);

        release_graph_tensor(input_tensor);

        /* output */
        tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
        set_node_output_tensor(pool_node, 0, output_tensor, TENSOR_TYPE_VAR);

        release_graph_tensor(output_tensor);

        /* attr */
        set_node_attr_int(pool_node, "kernel_h", &kernel_h);
        set_node_attr_int(pool_node, "kernel_w", &kernel_w);
        set_node_attr_int(pool_node, "stride_h", &stride_h);
        set_node_attr_int(pool_node, "stride_w", &stride_w);
        set_node_attr_int(pool_node, "pad_h0", &pad_h0);
        set_node_attr_int(pool_node, "pad_w0", &pad_w0);
        set_node_attr_int(pool_node, "pad_h1", &pad_h1);
        set_node_attr_int(pool_node, "pad_w1", &pad_w1);
        set_node_attr_int(pool_node, "alg", &method);

        release_graph_node(pool_node);

        return 0;
    }   
    

    graph_t create_pool_graph(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1,
                          int pad_w1, int m)
    {
        graph_t graph = create_graph(nullptr, nullptr, nullptr);

        if(graph == nullptr)
        {
            std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
            return nullptr;
        }

        const char* input_name = "data";
        const char* pool_name = "pool";

        if(create_input_node(graph, input_name, c, h, w) < 0)
        {
            std::cerr << "create input failed\n";
            return nullptr;
        }

        if(create_pool_node(graph, pool_name, input_name, k_h, k_w, s_h, s_w, pad_h0, pad_w0, pad_h1, pad_w1, m) < 0)
        {
            std::cerr << "create pool node failed\n";
            return nullptr;
        }

        /* set input/output node */

        const char* inputs[] = {input_name};
        const char* outputs[] = {pool_name};

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

    void get_input_data(float* dat,int nums)
    {
        for(int i = 0; i < nums; i++)
        {
            dat[i] = random() * 1000.0f;
        }
    }

    int test_pool(int c, int h, int w, int k_h, int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1,
              int m,int run_count)
    {
        graph_t graph = create_pool_graph(c, h, w, k_h, k_w, s_h, s_w, pad_h0, pad_w0, pad_h1, pad_w1, m);
        if(graph == nullptr)
            return 1;

        tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

        int ret = 0;
        if(prerun_graph(graph) < 0)
        {
            std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
            return 1;
        }

        release_graph_tensor(input_tensor);

        postrun_graph(graph);
        destroy_graph(graph);

        return ret;

    }

}

extern "C" int tengine_authed_test()
{
    if( AuthHelp::test_pool(1, 56, 56, 3, 3, 1, 1, 0, 0, 0, 0, 1,1) < 0 )
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

