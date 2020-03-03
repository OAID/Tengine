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
 * Author: chunyinglv@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_config.hpp"

// nasnet_mobile.pb download form:
// https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz
// model in: tengine-Server:/home/public/tf_models/nasnet_mobile

const char* model_file = "./models/nasnet_mobile.pb";
const char* label_file = "./models/synset_words.txt";
const char* image_file = "./tests/images/bike.jpg";

int img_h = 224;
int img_w = 224;
const float channel_mean[3] = {0, 0, 0};
float scale = 1. / 255;
using namespace TEngine;

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}
void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    image im = imread(image_file);

    image resImg = resize_image(im, img_w, img_h);

    int index = 0;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[index] = scale * (resImg.data[c * img_h * img_w + h * img_w + w] - mean[c]);
                index++;
            }
        }
    }
}
static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

int main(int argc, char* argv[])
{
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
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, channel_mean, scale);

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }

    /* run the graph */
    prerun_graph(graph);

    run_graph(graph, 1);

    // tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    std::string output_tensor_name = "final_layer/predictions";
    tensor_t output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }
    int dim_size = get_tensor_shape(output_tensor, dims, 4);
    if(dim_size < 0)
    {
        printf("get output tensor shape failed\n");
        return -1;
    }
    std::cout << "dim size is :" << dim_size << "\n";
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
        std::cout << labels[idx - 1] << "\"\n";
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
