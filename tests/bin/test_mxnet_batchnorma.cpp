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
 * Author: cmeng@openailab.com
 */
#include <cstdlib>
#include <memory>
#include "tengine_operations.h"
#include "tengine_c_api.h"

#include <cstdio>
#include <sstream>
#include <iostream>

int main(int argc, char* argv[])
{
    // load config
    const char* json_path = "./models/F0.1.3.05/model-faceid-F0.1.3.05-nol2.json";
    const char* param_path = "./models/F0.1.3.05/model-0039.params";
    const char* img_path = "./tests/images/feature5.png";
    if(argc > 3)
    {
        json_path = argv[1];
        param_path = argv[2];
        img_path = argv[3];
    }

    // set mean and scale
    float mean[] = {128.0, 128.0, 128.0};
    float default_scales[] = {0.0078125, 0.0078125, 0.0078125};

    // load img
    int img_h = 112;
    int img_w = 112;
    int img_c = 3;
    int img_hw = img_h * img_w;
    int img_size = img_h * img_w * img_c;
    int dims[] = {1, img_c, img_h, img_w};
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    image img = imread(img_path);
    image img2 = resize_image(img, img_h, img_w);
    float* img_data = ( float* )img2.data;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < img_c; c++)
            {
                input_data[c * img_hw + h * img_w + w] = ((*img_data - mean[c]) * default_scales[c]);
                img_data++;
            }
        }
    }

    // tengine init
    int ret = init_tengine();
    if(ret != 0)
    {
        std::cout << "init_tengine error, error code: " << get_tengine_errno() << std::endl;
        return -1;
    }
    std::cout << "get version: " << get_tengine_version() << std::endl;

    // create graph
    graph_t graph = create_graph(nullptr, "mxnet", json_path, param_path);
    if(graph == nullptr)
    {
        std::cout << "create_graph error: " << get_tengine_errno() << std::endl;
        return -1;
    }

    // create tensor
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::cout << "create input_tensor error: " << get_tengine_errno() << std::endl;
        return -1;
    }

    // set shape
    set_tensor_shape(input_tensor, dims, sizeof(float));

    // prerun
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun_graph error: " << get_tengine_errno() << std::endl;
        return -1;
    }
    // dump_graph(graph);
    // set_tensor_buffer
    set_tensor_buffer(input_tensor, input_data, img_size);

    // run predict
    if(run_graph(graph, 1) < 0)
    {
        std::cout << "run_graph error: " << get_tengine_errno() << std::endl;
        return -1;
    }

    // get output
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
    std::cout << "data_size: " << data_size << std::endl;
    // return -1;

    float* data = ( float* )get_tensor_buffer(output_tensor);
    for(int i = 0; i < data_size; i++)
    {
        std::cout << data[i] << ", ";
    }
    std::cout << std::endl;

    // release source
    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
