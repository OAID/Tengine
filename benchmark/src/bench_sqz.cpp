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
#include "tengine_cpp_api.h"
#include "common_util.hpp"

const char* model_file = "./models/squeezenet.tmfile";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

using namespace TEngine;

int repeat_count = 100;
void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    if(labels.is_open())
    {
        while(std::getline(labels, line))
            result.push_back(line);
    }        
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
        if(labels.size())
            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
        else
            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - " << idx << "\n";
        
    }
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
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

    tengine::Net squeezenet;
    tengine::Tensor input_tensor;
    tengine::Tensor output_tensor;

    /* load model */
    squeezenet.load_model(NULL, "tengine", model_file);

    /* prepare input data */
    input_tensor.create(img_w, img_h, 3);
    get_input_data(image_file, (float* )input_tensor.data, img_h, img_w, channel_mean, 1);
    
    /* forward */
    squeezenet.input_tensor("data", input_tensor);

    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)   
        squeezenet.run();

    unsigned long end_time = get_cur_time();
    unsigned long off_time = end_time - start_time;    

    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count, off_time);

    /* get result */
    squeezenet.extract_tensor("prob", output_tensor);

    /* after process */
    PrintTopLabels(label_file, (float*)output_tensor.data);
    
    std::cout << "--------------------------------------\n";
    std::cout << "ALL TEST DONE\n";

    return 0;
}
