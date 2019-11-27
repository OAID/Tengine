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
 * Author: haitao@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>

#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "common_util.hpp"

const char* text_file = "./models/sqz.prototxt";
const char* model_file = "./models/squeezenet_v1.1.caffemodel";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

using namespace TEngine;

int repeat_count = 100;
void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void PrintTopLabels(const char* label_file, float* data)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float* end = data + 1000;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
#if 1
    image img = imread(image_file);

    image resImg = resize_image(img, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);
    float* img_data = ( float* )resImg.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
#endif
}

int main(int argc, char* argv[])
{
    std::string device;
    char* cpu_list_str = nullptr;
    ;

    int res;

    while((res = getopt(argc, argv, "p:d:r:")) != -1)
    {
        switch(res)
        {
            case 'p':
                cpu_list_str = optarg;
                break;
            case 'd':
                device = optarg;
                break;
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    int img_h = 227;
    int img_w = 227;

    std::string model_name = "squeeze_net";

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);

    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);

    if(cpu_list_str)
        set_cpu_list(cpu_list_str);

    init_tengine();

    std::cout << "run-time library version: " << get_tengine_version() << "\n";

    if(request_tengine_version("0.9") < 0)
        return -1;

    graph_t graph = create_graph(nullptr, "caffe", text_file, model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
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

    int dims[] = {1, 3, img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    if(!device.empty())
        set_graph_device(graph, device.c_str());

    /* run the graph */

    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    // warm up
    run_graph(graph, 1);

    printf("REPEAT COUNT= %d\n", repeat_count);
    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;

    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
                off_time);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* data = ( float* )get_tensor_buffer(output_tensor);
    PrintTopLabels(label_file, data);
    std::cout << "--------------------------------------\n";

    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);

    postrun_graph(graph);
    destroy_graph(graph);

    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
