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
#include <vector>
#include <opencv2/opencv.hpp>

#include "prof_utils.hpp"
#include "tengine_c_api.h"
#include "tengine_config.hpp"

#define RUN_TIME 5000

using namespace std;

const char* image_file = "./tools/data/images.txt";
const char* label_file = "./tools/data/label.txt";

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

using namespace TEngine;

void get_input_data(std::string& image_file, float* input_data, int img_h, int img_w, float* mean, float* scale)
{
    cv::Mat sample = cv::imread(image_file, -1);
    if(sample.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
    cv::Mat img;
    if(sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if(sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = *img_data - mean[c];
                input_data[c * hw + h * img_w + w] = input_data[c * hw + h * img_w + w] * scale[c];
                img_data++;
            }
        }
    }
}

void split(std::string& s, std::string delim, std::vector<std::string>* ret)
{
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while(index != string::npos)
    {
        ret->push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if(index - last > 0)
    {
        ret->push_back(s.substr(last, index - last));
    }
}

void LoadImageFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream images(fname);

    std::string line;
    while(std::getline(images, line))
        result.push_back(line);
}

void LoadLableFile(std::vector<int>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
    {
        int lable_index = atoi(line.c_str());
        result.push_back(lable_index);
    }
}

int main(int argc, char* argv[])
{
    // Load image file
    std::vector<std::string> images;
    LoadImageFile(images, image_file);
    // Load Label file
    std::vector<int> labels;
    LoadLableFile(labels, label_file);

    /*
     1. prototxt
     2. mobile
     3. mean
     4. scale
     5. input tensor
     6. output tensor
     7. input weight and input height
     8. image_path
    */
    if(argc < 9)
    {
        std::cout << "The params are not enough \n";
    }

    std::string text_file = argv[1];
    std::string model_file = argv[2];
    std::string mean_val = argv[3];
    std::string scale_val = argv[4];
    std::string input_tensor_name = argv[5];
    std::string output_tensor_name = argv[6];
    std::string image_wh = argv[7];
    std::string val_file_path = argv[8];

    float means[3];
    float scales[3];
    string delima = ",";

    std::vector<string> res0;
    std::vector<string> res1;
    std::vector<string> wh;

    split(mean_val, delima, &res0);
    split(scale_val, delima, &res1);
    split(image_wh, delima, &wh);

    for(int i = 0; i < 3; i++)
    {
        means[i] = atof(res0[i].c_str());
        scales[i] = atof(res1[i].c_str());
    }

    int img_w = atoi(wh[0].c_str());
    int img_h = atoi(wh[1].c_str());

    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "caffe", text_file.c_str(), model_file.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, input_tensor_name.c_str());

    if(input_tensor == nullptr)
    {
        std::printf("cannot find tensor: %s\n", input_tensor_name.c_str());
        return -1;
    }

    int dims[] = {1, 3, img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);

    tensor_t output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());

    /* setup output buffer */

    void* output_data = malloc(sizeof(float) * 1000);

    if(set_tensor_buffer(output_tensor, output_data, 4 * 1000))
    {
        std::printf("set buffer for tensor: %s failed\n", output_tensor_name.c_str());
        return -1;
    }

    prerun_graph(graph);

    int top1 = 0;
    int top5 = 0;

    for(int i = 0; i < RUN_TIME; i++)
    {
        std::cout << "---------------------- " << i << " --------------------------------\n";
        /* prepare input data */
        std::string image_path = val_file_path + images[i];
        get_input_data(image_path, input_data, img_h, img_w, means, scales);

        set_tensor_buffer(input_tensor, input_data, img_size * 4);

        run_graph(graph, 1);

        int count = get_tensor_buffer_size(output_tensor) / 4;
        float* data = ( float* )(output_data);
        float* end = data + count;

        std::vector<float> result(data, end);
        std::vector<int> top_N = Argmax(result, 5);

        int right_idx = labels[i];
        if(right_idx == top_N[0])
        {
            top1++;
        }

        for(int j = 0; j < 5; j++)
        {
            if(right_idx == top_N[j])
            {
                top5++;
                break;
            }
        }
    }

    printf("top1 : %f\n", ( float )top1 / RUN_TIME);
    printf("top5 : %f\n", ( float )top5 / RUN_TIME);

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    postrun_graph(graph);
    destroy_graph(graph);

    free(output_data);
    free(input_data);

    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
