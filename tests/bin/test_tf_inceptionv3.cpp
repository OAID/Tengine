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
#include <opencv2/opencv.hpp>

#include "tengine_c_api.h"
#include "common_util.hpp"
#include "image_process.hpp"
#include "tengine_config.hpp"

//#define MOBILE_NET

const char* model_file = "./models/inception_v3_2016_08_28_frozen.pb";
const char* label_file = "./models/imagenet_slim_labels.txt";
const char* image_file = "./tests/images/grace_hopper.jpg";

int img_h = 299;
int img_w = 299;
float input_mean = 0;
float input_std = 255;

using namespace TEngine;

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

float* ReadImageFile(const std::string& image_file, cv::Mat& img, const int input_height, const int input_width,
                     const float input_mean, const float input_std)
{
    // Read image
    cv::Mat frame = cv::imread(image_file);
    if(!frame.data)
    {
        std::cout << "failed to read image file: " << image_file << std::endl;
        return nullptr;
    }

    // Convert BGR to RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::Mat img_resized;
    cv::Size input_geometry = cv::Size(input_width, input_height);
    cv::resize(frame, img_resized, input_geometry);

    // Convert to float 32, channel 3
    img_resized.convertTo(img_resized, CV_32FC3);

    img = (img_resized - input_mean) / input_std;

    std::vector<cv::Mat> input_channels;
    float* input_data = ( float* )std::malloc(input_height * input_width * 3 * 4);
    float* ptr = input_data;

    for(int i = 0; i < 3; ++i)
    {
        cv::Mat channel(input_height, input_width, CV_32FC1, ptr);
        input_channels.push_back(channel);
        ptr += input_height * input_width;
    }

    cv::split(img, input_channels);

    return input_data;
}

int main(int argc, char* argv[])
{
    /* prepare input data */
    cv::Mat img;
    float* input_data = ReadImageFile(image_file, img, img_h, img_w, input_mean, input_std);
    ;

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

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */
    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }

    /* run the graph */
    prerun_graph(graph);

    run_graph(graph, 1);

    // const char * output_tensor_name="InceptionV3/Predictions/Softmax";
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
