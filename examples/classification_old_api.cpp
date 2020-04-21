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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: jxyang@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <string.h>
#include <iomanip>
#include <algorithm>
#include "tengine_c_api.h"
#include "tengine_operations.h"
#include "common.hpp"

#define DEFAULT_MODEL_NAME "squeezenet"
#define DEFAULT_LABEL_FILE "./synset_words.txt"
#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_REPEAT_CNT 1

static std::string gExcName{""};

bool run_tengine_library(const char* model_name, const char* tm_file, const char* label_file, const char* image_file,
                         int img_h, int img_w, const float* mean, float scale, int repeat_count)
{
    // init
    init_tengine();
    std::cout << "tengine library version: " << get_tengine_version() << "\n";
    if(request_tengine_version("1.0") < 0)
        return false;

    // create graph
    graph_t graph = create_graph(nullptr, "tengine", tm_file);
    if(graph == nullptr)
    {
        std::cerr << "Create graph failed.\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return false;
    }

    // set input
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};
    if( strstr(model_name, "_tflow") != 0 || strstr(model_name, "_tflite") != 0)
    {
        dims[1] = img_h;
        dims[2] = img_w;
        dims[3] = 3;
    }
    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    uint8_t* input_data_tflite = ( uint8_t* )malloc(img_size);
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::cerr << "Get input tensor failed\n";
        return false;
    }
    set_tensor_shape(input_tensor, dims, 4);
    if(strstr(model_name, "_tflite") != 0){
        int val = 1;
        set_graph_attr(graph, "low_mem_mode", &val, sizeof(val));
    }

    // prerun
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "Prerun graph failed\n";
        return false;
    }

    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    std::string str = model_name;
    std::string::size_type p1 = str.find("_tflow");
    std::string::size_type p2 = str.find("_mx");
    std::string::size_type p3 = str.find("_tflite");    
    if(p1 != std::string::npos){
        get_input_data_tf(image_file, input_data, img_h, img_w, mean, scale);
    }
    else if(p2 != std::string::npos){
        get_input_data_mx(image_file, input_data, img_h, img_w, mean);
    }
    else if(p3 != std::string::npos){
        get_input_data_uint8(image_file, input_data_tflite, img_h, img_w);
    }    
    else{
        get_input_data(image_file, input_data, img_h, img_w, mean, scale);
    }

    if(strstr(model_name, "_tflite") != 0){
        set_tensor_buffer(input_tensor, input_data_tflite, img_size * 4);
    }
    else
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
    
    // warm up
    run_graph(graph, 1);
    run_graph(graph, 1);
    
    // run
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return false;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "\nModel name : " << model_name << "\n"
              << "tengine model file : " << tm_file << "\n"
              << "label file : " << label_file << "\n"
              << "image file : " << image_file << "\n"
              << "img_h, imag_w, scale, mean[3] : " << img_h << " " << img_w << " " << scale << " " << mean[0] << " "
              << mean[1] << " " << mean[2] << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";
    std::cout << "--------------------------------------\n";

    // print output
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
  
    if(strstr(model_name, "_tflite") != 0){
        uint8_t* data = ( uint8_t* )get_tensor_buffer(output_tensor);
        int data_size = get_tensor_buffer_size(output_tensor);
        float output_scale = 0.0f;
        int zero_point = 0;
        get_tensor_quant_param(output_tensor, &output_scale, &zero_point, 1);
        PrintTopLabels_uint8(label_file, data, data_size, output_scale, zero_point);
    } 
    else if (strstr(model_name, "nasnet_tflow") != 0)
    {
        std::string output_tensor_name = "final_layer/predictions";
        output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());
        float* data = ( float* )get_tensor_buffer(output_tensor);
        int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
        float* end = data + data_size;

        std::vector<float> result(data, end);
        std::vector<int> top_N = Argmax(result, 5);
        std::vector<std::string> labels;

        LoadLabelFile_nasnet(labels, label_file);

        for(unsigned int i = 0; i < top_N.size(); i++)
        {
            int idx = top_N[i];

            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
            std::cout << labels[idx-1] << "\"\n";
        }        
    }
    else {
        float* data = ( float* )get_tensor_buffer(output_tensor);
        int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
        PrintTopLabels_common(label_file, data, data_size, model_name);
    }
    std::cout << "--------------------------------------\n";

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();

    return true;
}

void show_usage()
{
    std::cout << "[Usage]: " << gExcName << " [-h]\n"
              << "    [-m model_file] [-l label_file] [-i image_file]\n"
              << "    [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";

    std::cout << "\nmobilenet example: \n" << "    ./classification -m /path/to/mobilenet.tmfile -l /path/to/labels.txt -i /path/to/img.jpg -g 224,224 -s 0.017 -w 104.007,116.669,122.679" << std::endl;
}

int main(int argc, char* argv[])
{
    gExcName = std::string(argv[0]);
    int repeat_count = DEFAULT_REPEAT_CNT;
    std::string model_name;
    std::string tm_file;
    std::string label_file;
    std::string image_file;
    std::vector<int> hw;
    std::vector<float> ms;
    int img_h = 0;
    int img_w = 0;
    float scale = 0.0;
    float mean[3] = {-1.0, -1.0, -1.0};

    int res;
    while((res = getopt(argc, argv, "m:n:t:l:i:g:s:w:r:h")) != -1)
    {
        switch(res)
        {
            case 'm':
                tm_file = optarg;
                break;
            case 'l':
                label_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'g':
                hw = ParseString<int>(optarg);
                if(hw.size() != 2)
                {
                    std::cerr << "Error -g parameter.\n";
                    show_usage();
                    return -1;
                }
                img_h = hw[0];
                img_w = hw[1];
                break;
            case 's':
                scale = strtof(optarg, NULL);
                break;
            case 'w':
                ms = ParseString<float>(optarg);
                if(ms.size() != 3)
                {
                    std::cerr << "Error -w parameter.\n";
                    show_usage();
                    return -1;
                }
                mean[0] = ms[0];
                mean[1] = ms[1];
                mean[2] = ms[2];
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    if (tm_file.empty())
    {
        std::cerr << "Error: Tengine model file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    if(image_file.empty())
    {
        std::cerr << "Error: Image file not specified!" << std::endl;
        show_usage();
        return -1;
    }
    if(label_file.empty())
    {
        label_file = DEFAULT_LABEL_FILE;
        std::cout << "Label file not specified, use default [" << label_file << "]." << std::endl;
    }
    // check input files
    if(!check_file_exist(tm_file) || !check_file_exist(label_file) || !check_file_exist(image_file))
        return -1;

    if(img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        std::cout << "Image height not specified, use default [" << DEFAULT_IMG_H << "]" << std::endl;
    }
    if(img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        std::cout << "Image width not specified, use default [" << DEFAULT_IMG_W << "]" << std::endl;
    }
    if(scale == 0.0)
    {
        scale = DEFAULT_SCALE;
        std::cout << "Scale value not specified, use default [" << scale << "]" << std::endl;
    }
    if(mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        std::cout << "Mean value not specified, use default [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]" << std::endl;
    }
    if(model_name.empty())
        model_name = tm_file;
    
    // start to run
    if(!run_tengine_library(model_name.c_str(), tm_file.c_str(), label_file.c_str(), image_file.c_str(), img_h, img_w,
                            mean, scale, repeat_count))
        return -1;

    std::cout << "ALL TEST DONE\n";

    return 0;
}

