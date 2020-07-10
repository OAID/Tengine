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
 * Author: cmeng@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <float.h>

#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_cpp_api.h"
#include "common_util.hpp"

using namespace TEngine;

typedef struct s_MODEL_CONFIG
{
    const char* model_file;
    int img_h;
    int img_w;
} S_MODEL_CONFIG;

int debug = 0;
int repeat_count = 10;
int warm_count = 3;

void benchmark_graph(const char* graph_name, const std::string model_file, int img_h, int img_w, int c, int n)
{
    tengine::Net net;
    tengine::Tensor input_tensor;
    tengine::Tensor output_tensor;

    /* load model */
    net.load_model(NULL, "tengine", model_file.c_str());
    if(debug)
        net.dump();

    /* prepare input data */
    input_tensor.create(n, img_w, img_h, c);
    input_tensor.fill(0.1f);

    /* forward */
    net.input_tensor(0, 0, input_tensor);

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for(int i=0; i < warm_count; i++)
	    net.run();
    
    for(int i = 0; i < repeat_count; i++)
    {
        unsigned long start_time = get_cur_time();

        net.run();

        unsigned long end_time = get_cur_time();
        double time = end_time - start_time;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= repeat_count;

    fprintf(stderr, "%20s  min = %7.2f ms   max = %7.2f ms   avg = %7.2f ms\n", graph_name, time_min / 1000,
            time_max / 1000, time_avg / 1000);

    std::cout << "--------------------------------------\n";
}

int main(int argc, char* argv[])
{
    std::string file_path = "";
    char* cpu_list_str = nullptr;
    int res;
    int select_num = -1;
	
    while((res = getopt(argc, argv, "p:r:d:s:")) != -1)
    {
        switch(res)
        {
            case 'p':
                cpu_list_str = optarg;
                break;
            case 'd':
                debug = *optarg;
                break;
            case 's':
                select_num = strtoul(optarg, NULL, 10);
                break;
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;

            default:
                break;
        }
    }
    tengine::Net::Init();
    switch(select_num)
    {
        case 0:
            benchmark_graph("mobilenetv1", "./models/mobilenet_benchmark.tmfile", 224, 224, 3, 1);
            break;
        case 1:
            benchmark_graph("squeezenet_v1.1", "./models/squeezenet_v1.1_benchmark.tmfile", 227, 227, 3, 1);
            break;
        case 2:
            benchmark_graph("vgg16", "./models/vgg16_benchmark.tmfile", 224, 224, 3, 1);
            break;
        case 3:
            benchmark_graph("mssd", "./models/mssd_benchmark.tmfile", 300, 300, 3, 1);
            break;
        case 4:
            benchmark_graph("resnet50", "./models/resnet50_benchmark.tmfile", 224, 224, 3, 1);
            break;
        case 5:
            benchmark_graph("retinaface", "./models/retinaface_benchmark.tmfile", 1024, 1024, 3, 1);
            break;
        case 6:
            benchmark_graph("yolov3", "./models/yolov3_benchmark.tmfile", 416, 416, 3, 1);
            break;
        case 7:
            benchmark_graph("mobilenetv2", "./models/mobilenet_v2_benchmark.tmfile", 224, 224, 3, 1);
            break;
	    case 8:
            benchmark_graph("mobilenetv3", "./models/mobilenetv3_benchmark.tmfile", 224, 224, 3, 1);
            break;
        case 9:
            benchmark_graph("mobilefacenets",   "./models/mobilefacenets_benchmark.tmfile",     112, 112, 3, 1);
            break;
        default:
            benchmark_graph("squeezenet_v1.1",  "./models/squeezenet_v1.1_benchmark.tmfile",    227, 227, 3, 1);
            benchmark_graph("mobilenetv1",      "./models/mobilenet_benchmark.tmfile",          224, 224, 3, 1);
            benchmark_graph("mobilenetv2",      "./models/mobilenet_v2_benchmark.tmfile",       224, 224, 3, 1);
            benchmark_graph("mobilenetv3",      "./models/mobilenet_v3_benchmark.tmfile",       224, 224, 3, 1);
            benchmark_graph("shufflenetv2",     "./models/shufflenet_v2_benchmark.tmfile",      224, 224, 3, 1);
            benchmark_graph("resnet18",         "./models/resnet18_benchmark.tmfile",           224, 224, 3, 1);
            benchmark_graph("resnet50",         "./models/resnet50_benchmark.tmfile",           224, 224, 3, 1);
            benchmark_graph("googlenet",        "./models/googlenet_benchmark.tmfile",          224, 224, 3, 1);
            benchmark_graph("inceptionv3",      "./models/inception_v3_benchmark.tmfile",       299, 299, 3, 1);
            benchmark_graph("vgg16",            "./models/vgg16_benchmark.tmfile",              224, 224, 3, 1);
            benchmark_graph("mssd",             "./models/mssd_benchmark.tmfile",               300, 300, 3, 1);
            benchmark_graph("retinaface",       "./models/retinaface_benchmark.tmfile",         320, 240, 3, 1);
            benchmark_graph("yolov3",           "./models/yolov3_benchmark.tmfile",             416, 416, 3, 1);
            benchmark_graph("yolov3_tiny",      "./models/yolov3_tiny_benchmark.tmfile",        416, 416, 3, 1);
            benchmark_graph("mobilefacenets",   "./models/mobilefacenets_benchmark.tmfile",     112, 112, 3, 1);
    }
    tengine::Net::Deinit();
    std::cout << "ALL TEST DONE\n";

    return 0;
}
