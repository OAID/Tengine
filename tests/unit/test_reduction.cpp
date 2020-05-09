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
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <math.h>

#include "tengine_c_api.h"

std::string model_name = "../models/reduce_mean_pb.tmfile";

float result = 6.224632;

int main(int argc, char* argv[])
{
    init_tengine();

    graph_t graph = create_graph(nullptr, "tengine", model_name.c_str());
   
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return 1;
    }

    set_graph_layout(graph, TENGINE_LAYOUT_NHWC);
    

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dim[4] = {1, 28, 28,1};

    set_tensor_shape(input_tensor, dim, 4);

    int input_size = get_tensor_buffer_size(input_tensor);
    float* input_data = ( float* )malloc(input_size);

    for(int i = 0; i < input_size/sizeof(float); i++){
        input_data[i] = i * 0.1  - ( i%2) ;
    }
    set_tensor_buffer(input_tensor, input_data, input_size);

    prerun_graph(graph);


    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    int dim_size = get_tensor_shape(output_tensor, dim, 4);
    if(dim_size < 0)
    {
        printf("Get output tensor shape failed\n");
        return -1;
    }
    int size = 1;

    for(int i = 0; i < dim_size; i++){
        size *= dim[i];
    }    
    float* data = ( float* )(get_tensor_buffer(output_tensor));
    
    for(int i = 0; i < size; i++)
    {    
        
        float tmp = fabs(data[i]) - fabs(result);
        if(tmp > 0.01){
            printf("Test Failed \n");
            return 0;
        }
       //printf("%f ", data[i]);
    }
    printf("pass\n");
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
