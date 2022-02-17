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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 * Update: lswang@openailab.com
 */

#include "tengine/c_api.h"

#include "common/cmdline.hpp"
#include "common/timer.hpp"

#include <cstdio>
#include <string>


int benchmark_loop = 1;
int benchmark_threads = 1;
int benchmark_model = -1;
int benchmark_cluster = 0;
int benchmark_mask = 0xFFFF;
int benchmark_data_type = 0;
std::string benchmark_device = "";
context_t s_context;


int get_tenser_element_size(int data_type)
{
    switch (data_type)
    {
    case TENGINE_DT_FP32:
    case TENGINE_DT_INT32:
        return 4;
    case TENGINE_DT_FP16:
    case TENGINE_DT_INT16:
        return 2;
    case TENGINE_DT_INT8:
    case TENGINE_DT_UINT8:
        return 1;
    default:
        return 0;
    }
}


int benchmark_graph(options_t* opt, const char* name, const char* file, int height, int width, int channel, int batch)
{
    // create graph, load tengine model xxx.tmfile
    graph_t graph = create_graph(s_context, "tengine", file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Tengine Benchmark: Create graph failed.\n");
        return -1;
    }

    // set the input shape to initial the graph, and pre-run graph to infer shape
    int input_size = height * width * channel;
    int shape[] = { batch, channel, height, width };    // nchw

    std::vector<unsigned char> input_buffer(batch * input_size * get_tenser_element_size(benchmark_data_type));

    memset(input_buffer.data(), 1, input_buffer.size());

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Tengine Benchmark: Get input tensor failed.\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, shape, 4) < 0)
    {
        fprintf(stderr, "Tengine Benchmark: Set input tensor shape failed.\n");
        return -1;
    }

    if (prerun_graph_multithread(graph, *opt) < 0)
    {
        fprintf(stderr, "Tengine Benchmark: Pre-run graph failed.\n");
        return -1;
    }

    // prepare process input data, set the data mem to input tensor
    if (set_tensor_buffer(input_tensor, input_buffer.data(), (int)input_buffer.size()) < 0)
    {
        fprintf(stderr, "Tengine Benchmark: Set input tensor buffer failed\n");
        return -1;
    }

    // warning up graph
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Tengine Benchmark: Run graph failed.\n");
        return -1;
    }

    std::vector<float> time_cost(benchmark_loop);

    // run graph
    for (int i = 0; i < benchmark_loop; i++)
    {
        Timer timer;
        int ret = run_graph(graph, 1);
        time_cost[i] = timer.TimeCost();

        if (0 != ret)
        {
            fprintf(stderr, "Tengine Benchmark: Run graph failed\n");
            return -1;
        }
    }

    float min = time_cost[0], max = time_cost[0], sum = 0.f;
    for (const auto& var : time_cost)
    {
        if (min > var)
        {
            min = var;
        }

        if (max < var)
        {
            max = var;
        }

        sum += var;
    }
    sum /= (float)time_cost.size();


    fprintf(stderr, "%20s  min = %7.2f ms   max = %7.2f ms   avg = %7.2f ms\n", name, min, max, sum);

    // release tengine graph
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    return 0;
}


int main(int argc, char* argv[])
{
    cmdline::parser cmd;

    cmd.add<int>("loop_count", 'r', "benchmark loops count", false, 1);
    cmd.add<int>("thread_count", 't', "benchmark threads count", false, 1);
    cmd.add<int>("cpu_cluster", 'p', "cpu cluster [0:auto, 1:big, 2:middle, 3:little]", false, 0);
    cmd.add<int>("model", 's', "benchmark which model, \"-1\" means all models", false, -1);
    cmd.add<int>("cpu_mask", 'a', "benchmark on masked cpu core(s)", false, -1);
    cmd.add<std::string>("device", 'd', "device name (should be upper-case)", false);
    cmd.add<std::string>("model_file", 'm', "path to a model file", false);
    cmd.add<std::string>("input_shape", 'i', "shape of input (n,c,h,w)", false);
    cmd.add<int>("input_dtype", 'f', "data type of input", false);

    cmd.parse_check(argc, argv);

    benchmark_loop = cmd.get<int>("loop_count");
    benchmark_threads = cmd.get<int>("thread_count");
    benchmark_model = cmd.get<int>("model");
    benchmark_cluster = cmd.get<int>("cpu_cluster");
    benchmark_mask = cmd.get<int>("cpu_mask");
    benchmark_device = cmd.get<std::string>("device");
    std::string benchmark_model_file = cmd.get<std::string>("model_file");
    std::string input_shape = cmd.get<std::string>("input_shape");
    benchmark_data_type = cmd.get<int>("input_dtype");
    if (benchmark_device.empty())
    {
        benchmark_device = "CPU";
    }
    else
    {
        for (int i = 0; i < benchmark_device.length(); i++)
        {
            benchmark_device[i] = ::toupper(benchmark_device[i]);
        }
    }

    fprintf(stdout, "Tengine benchmark:\n");
    fprintf(stdout, "  loops:    %d\n", benchmark_loop);
    fprintf(stdout, "  threads:  %d\n", benchmark_threads);
    fprintf(stdout, "  cluster:  %d\n", benchmark_cluster);
    fprintf(stdout, "  affinity: 0x%X\n", benchmark_mask);
    fprintf(stdout, "  device:   %s\n", benchmark_device.c_str());

    // initialize tengine
    if (0 != init_tengine())
    {
        fprintf(stderr, "Tengine Benchmark: Initialize tengine failed.\n");
        return -1;
    }
    fprintf(stdout, "Tengine-lite library version: %s\n", get_tengine_version());

    s_context = create_context("ctx", benchmark_device.empty() ? 0 : 1);
    if (!benchmark_device.empty())
    {
        int ret = set_context_device(s_context, benchmark_device.c_str(), nullptr, 0);
        if (0 != ret)
        {
            fprintf(stderr, "Set context device failed: %d.\n", ret);
            return false;
        }
    }

    struct options opt;
    opt.num_thread = benchmark_threads;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = benchmark_mask;

    switch (benchmark_cluster)
    {
        case 0:
            opt.cluster = TENGINE_CLUSTER_ALL;
            break;
        case 1:
            opt.cluster = TENGINE_CLUSTER_BIG;
            break;
        case 2:
            opt.cluster = TENGINE_CLUSTER_MEDIUM;
            break;
        case 3:
            opt.cluster = TENGINE_CLUSTER_LITTLE;
            break;
        default:
            opt.cluster = TENGINE_CLUSTER_ALL;
    }

    // run benchmarks
    switch(benchmark_model)
    {
        case 0:
            benchmark_graph(&opt, "squeezenet_v1.1",  "./models/squeezenet_v1.1_benchmark.tmfile",    227, 227, 3, 1);
            break;
        case 1:
            benchmark_graph(&opt, "mobilenetv1",      "./models/mobilenet_benchmark.tmfile",          224, 224, 3, 1);
            break;
        case 2:
            benchmark_graph(&opt, "mobilenetv2",      "./models/mobilenet_v2_benchmark.tmfile",       224, 224, 3, 1);
            break;
        case 3:
            benchmark_graph(&opt, "mobilenetv3",      "./models/mobilenet_v3_benchmark.tmfile",       224, 224, 3, 1);
            break;
        case 4:
            benchmark_graph(&opt, "shufflenetv2",     "./models/shufflenet_v2_benchmark.tmfile",      224, 224, 3, 1);
            break;
        case 5:
            benchmark_graph(&opt, "resnet18",         "./models/resnet18_benchmark.tmfile",           224, 224, 3, 1);
            break;
        case 6:
            benchmark_graph(&opt, "resnet50",         "./models/resnet50_benchmark.tmfile",           224, 224, 3, 1);
            break;
        case 7:
            benchmark_graph(&opt, "googlenet",        "./models/googlenet_benchmark.tmfile",          224, 224, 3, 1);
            break;
        case 8:
            benchmark_graph(&opt, "inceptionv3",      "./models/inception_v3_benchmark.tmfile",       299, 299, 3, 1);
            break;
        case 9:
            benchmark_graph(&opt, "vgg16",            "./models/vgg16_benchmark.tmfile",              224, 224, 3, 1);
            break;
        case 10:
            benchmark_graph(&opt, "mssd",             "./models/mssd_benchmark.tmfile",               300, 300, 3, 1);
            break;
        case 11:
            benchmark_graph(&opt, "retinaface",       "./models/retinaface_benchmark.tmfile",         320, 240, 3, 1);
            break;
        case 12:
            benchmark_graph(&opt, "yolov3_tiny",      "./models/yolov3_tiny_benchmark.tmfile",        416, 416, 3, 1);
            break;
        case 13:
            benchmark_graph(&opt, "mobilefacenets",   "./models/mobilefacenets_benchmark.tmfile",     112, 112, 3, 1);
            break;
        default:
            if (!benchmark_model_file.empty()) {
                int n = 1, c = 3, h = 224, w = 224;
                if (!input_shape.empty()) {
                    char ch;
                    int count = sscanf(input_shape.c_str(), "%u%c%u%c%u%c%u", &n, &ch, &c, &ch, &h, &ch, &w);
                    if (count == 3) {
                        w = h, h = c, c = n, n = 1;
                    } else if (count == 2) {
                        w = c, h = n, c = 3, n = 1;
                    }
                }
                benchmark_graph(&opt, benchmark_model_file.c_str(), benchmark_model_file.c_str(), w, h, c, n);
                break;
            }
            benchmark_graph(&opt, "squeezenet_v1.1",  "./models/squeezenet_v1.1_benchmark.tmfile",    227, 227, 3, 1);
            benchmark_graph(&opt, "mobilenetv1",      "./models/mobilenet_benchmark.tmfile",          224, 224, 3, 1);
            benchmark_graph(&opt, "mobilenetv2",      "./models/mobilenet_v2_benchmark.tmfile",       224, 224, 3, 1);
            benchmark_graph(&opt, "mobilenetv3",      "./models/mobilenet_v3_benchmark.tmfile",       224, 224, 3, 1);
            benchmark_graph(&opt, "shufflenetv2",     "./models/shufflenet_v2_benchmark.tmfile",      224, 224, 3, 1);
            benchmark_graph(&opt, "resnet18",         "./models/resnet18_benchmark.tmfile",           224, 224, 3, 1);
            benchmark_graph(&opt, "resnet50",         "./models/resnet50_benchmark.tmfile",           224, 224, 3, 1);
            benchmark_graph(&opt, "googlenet",        "./models/googlenet_benchmark.tmfile",          224, 224, 3, 1);
            benchmark_graph(&opt, "inceptionv3",      "./models/inception_v3_benchmark.tmfile",       299, 299, 3, 1);
            benchmark_graph(&opt, "vgg16",            "./models/vgg16_benchmark.tmfile",              224, 224, 3, 1);
            benchmark_graph(&opt, "mssd",             "./models/mssd_benchmark.tmfile",               300, 300, 3, 1);
            benchmark_graph(&opt, "retinaface",       "./models/retinaface_benchmark.tmfile",         320, 240, 3, 1);
            benchmark_graph(&opt, "yolov3_tiny",      "./models/yolov3_tiny_benchmark.tmfile",        416, 416, 3, 1);
            benchmark_graph(&opt, "mobilefacenets",   "./models/mobilefacenets_benchmark.tmfile",     112, 112, 3, 1);
    }

    /* release tengine */
    destroy_context(s_context);
    release_tengine();
    fprintf(stderr, "ALL TEST DONE.\n");

    return 0;
}
