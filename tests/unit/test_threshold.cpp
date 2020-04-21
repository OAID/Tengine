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

const char* model_file = "../models/threshold.tmfile";


float result[27] = { 0.000000, 0.000000, 1.000000, 0.000000, 1.000000,
                     0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 
                     1.000000, 0.000000, 1.000000, 1.000000, 1.000000,
                      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
                      1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
                      1.000000, 1.000000};
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
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }
    int img_h = 3;
    int img_w = 3;
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);

    for(int i = 0; i < img_h*img_w*3; i++){
        input_data[i] = i * 0.1  - ( i%2) ;
    }
    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
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

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    
    
    for(int i = 0; i < size; i++)
    {
        
        float tmp = fabs(data[i]) - fabs(result[i]);
        if(tmp > 0.00001){
            printf("Test Failed \n");
            return 0;
        }
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
