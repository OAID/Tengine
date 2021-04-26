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


int benchmark_graph(options_t* opt, const char* name, const char* file, int height, int width, int channel, int batch)
{
    // create graph, load tengine model xxx.tmfile
    graph_t graph = create_graph(nullptr, "tengine", file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Tengine Benchmark: Create graph failed.\n");
        return -1;
    }

    // set the input shape to initial the graph, and pre-run graph to infer shape
    int input_size = height * width * channel;
    int shape[] = { batch, channel, height, width };    // nchw

    std::vector<float> inout_buffer(input_size);

    memset(inout_buffer.data(), 1, inout_buffer.size() * sizeof(float));

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
    if (set_tensor_buffer(input_tensor, inout_buffer.data(), (int)(inout_buffer.size() * sizeof(float))) < 0)
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

    cmd.parse_check(argc, argv);

    benchmark_loop = cmd.get<int>("loop_count");
    benchmark_threads = cmd.get<int>("thread_count");
    benchmark_model = cmd.get<int>("model");
    benchmark_cluster = cmd.get<int>("cpu_cluster");
    benchmark_mask = cmd.get<int>("cpu_mask");

    fprintf(stdout, "Tengine benchmark:\n");
    fprintf(stdout, "  loops:    %d\n", benchmark_loop);
    fprintf(stdout, "  threads:  %d\n", benchmark_threads);
    fprintf(stdout, "  cluster:  %d\n", benchmark_cluster);
    fprintf(stdout, "  affinity: 0x%X\n", benchmark_mask);

    // initialize tengine
    if (0 != init_tengine())
    {
        fprintf(stderr, "Tengine Benchmark: Initialize tengine failed.\n");
        return -1;
    }
    fprintf(stdout, "Tengine-lite library version: %s\n", get_tengine_version());

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
    release_tengine();
    fprintf(stderr, "ALL TEST DONE.\n");

    return 0;
}
