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
 * Copyright (c) 2017, Open AI Lab
 * Author: bzhang@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "common_util.hpp"
#include "image_process.hpp"

const char* model_file = "../models/roialign_mx.tmfile";


float input1[64] ={0.88, 0.44, 0.11, 0.22, 0.33, 0.12, 0.14, 0.42,
                0.78, 0.24, 0.14, 0.26, 0.32, 0.53, 0.62, 0.32,
                0.54, 0.35, 0.12, 0.34, 0.65, 0.43, 0.65, 0.43,
                0.65, 0.34, 0.65, 0.76, 0.87, 0.98, 0.34, 0.65,
                0.46, 0.42, 0.43, 0.54, 0.65, 0.02, 0.04, 0.06,
                0.68, 0.43, 0.15, 0.52, 0.35, 0.76, 0.43, 0.91,
                0.12, 0.94, 0.51, 0.32, 0.37, 0.92, 0.64, 0.52,
                0.89, 0.78, 0.81, 0.85, 0.78, 0.57, 0.46, 0.43};

float input2[4] = {2, 4, 7, 6};
float result[4] = {0.467778, 0.351250, 0.390833, 0.683750};
using namespace TEngine;

int main(int argc, char* argv[])
{
    /* prepare input data */

    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph, 1, 0);

    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }
    int img_h = 8;
    int img_w = 8;
    int dims[] = {1, 1, 8, 8};
    int dims2[] = {1,4};  
    //dump_graph(graph);
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_shape(input_tensor1, dims2, 2);  
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w );

    for(int i = 0; i < img_h*img_w; i++){
        input_data[i] = input1[i] ;
    }
    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_data, img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }
    float* input_data2 = ( float* )malloc(sizeof(float) * 4 );

    for(int i = 0; i < 4; i++){
        input_data2[i] = input2[i] ;
    }    
    if(set_tensor_buffer(input_tensor1, input_data2, 4 * sizeof(float)) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }
    /* run the graph */
    prerun_graph(graph);

    run_graph(graph, 1);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    if(output_tensor == nullptr)
    {
        std::printf("Cannot find output tensor , node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dim_size = get_tensor_shape(output_tensor, dims, 4);
    if(dim_size < 0)
    {
        printf("Get output tensor shape failed\n");
        return -1;
    }
    int size = 1;

    for(int i = 0; i < dim_size; i++){
        size *= dims[i];
    }

    int count = get_tensor_buffer_size(output_tensor) / 4;

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    
    
    for(int i = 0; i < size; i++)
    {
        
        
        float tmp = fabs(data[i]) - fabs(result[i]);
        if(tmp > 0.00001){
            printf("Test Failed \n");
            return 0;
        }
        //printf("%f ", data[i]);
    }
    printf("pass\n");
    
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
