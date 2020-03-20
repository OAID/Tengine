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

#include "tengine_c_api.h"
#include "common_util.hpp"
#include "tengine_operations.h"

const char* model_file = "./models/frozen_mobilenet_v1_224.pb";
const char* label_file = "./models/synset_words.txt";
const char* image_file = "./tests/images/cat.jpg";

int img_h = 224;
int img_w = 224;
float input_mean = -127;
float input_std = 127;

using namespace TEngine;
using namespace std;

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

float* ReadImageFile(const std::string& image_file, const int input_height, const int input_width,
                     const float input_mean, const float input_std)
{
    image img = imread(image_file.c_str());

    // Resize the image

    image resImg = resize_image(img, input_width, input_height);
    float* input_data = ( float* )malloc(3 * input_height * input_width * 4);
    for(int h = 0; h < input_height; h++)
    {
        for(int w = 0; w < input_width; ++w)
        {
            for(int c = 0; c < 3; c++)
            {
                int out_idx = h * input_width * 3 + w * 3 + c;
                int in_idx = c * input_height * input_width + h * input_width + w;
                input_data[out_idx] = (resImg.data[in_idx] - input_mean) / input_std;
            }
        }
    }

    return input_data;
}

int main(int argc, char* argv[])
{
    /* prepare input data */
    float* input_data = ReadImageFile(image_file, img_h, img_w, input_mean, input_std);
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tensorflow", model_file);
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // dump_graph(graph);

    /* set input shape */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d \n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, img_h, img_w, 3};
    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */
    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }

    /* run the graph */
    prerun_graph(graph);
    // dump_graph(graph);
    run_graph(graph, 1);

    // const char * output_tensor_name="MobilenetV1/Predictions/Softmax";
    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);

    int dim_size = get_tensor_shape(output_tensor, dims, 4);
    if(dim_size < 0)
    {
        printf("get output tensor shape failed\n");
        return -1;
    }

    printf("output tensor shape: [");
    for(int i = 0; i < dim_size; i++)
        printf("%d ", dims[i]);
    printf("]\n");

    int count = get_tensor_buffer_size(output_tensor) / 4;

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    float* end = data + count;

    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);
    std::vector<std::string> labels;

    LoadLabelFile(labels, label_file);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx] << "\"\n";
    }

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
