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
 * Author: bzhang@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "common.hpp"
#include "tengine_operations.h"

//ncnn optimizer mobilenet
const char* model_file = "./models/mobilenet_opt.bin";
const char* text_file = "./models/mobilenet_opt.param";

const char* image_file = "./models/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};


int repeat_count = 1;

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w)
{
    float means[3] = {104.007, 116.669, 122.679};
    float scales[3] = {0.017, 0.017, 0.017};
    //float scales[3] = {1, 1, 1};
    image img = imread(image_file, img_w, img_h, means, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*img.c*img_w*img_h); 
    free_image(img);
}


int main(int argc, char* argv[])
{
    int res;

    while((res = getopt(argc, argv, "r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    // const char * model_name="mobilenet";
    int n = 1;
    int c = 3;
    int h = 224;
    int w = 224;

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * h * w * 3);

    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return 1;

    //init the input data all 1.f
    get_input_data(image_file, input_data, h, w);

    
    printf("param file: %s \n", text_file);
    printf("model_file: %s \n", model_file);

    graph_t graph = create_graph(nullptr, "ncnn", text_file,model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
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

    int dims[] = {1, 3, h, w};

    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_data, 3 * h * w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    /* run the graph */
    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }
    // benchmark start here

    for(int i = 0; i < repeat_count; i++)
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

    printf("output tensor shape: [");

    for(int i = 0; i < dim_size; i++)
        printf("%d ", dims[i]);

    printf("]\n");

    //int count = get_tensor_buffer_size(output_tensor) / 4;
    float* data = ( float* )(get_tensor_buffer(output_tensor));

    int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    PrintTopLabels(label_file, data, data_size);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    postrun_graph(graph);

    destroy_graph(graph);

    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
