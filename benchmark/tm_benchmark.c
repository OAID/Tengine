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
 * Author: qtang@openailab.com
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "tengine_c_api.h"

#define DEFAULT_LOOP_COUNT      1
#define DEFAULT_THREAD_COUNT    1
#define DEFAULT_CLUSTER         TENGINE_CLUSTER_ALL

int loop_counts = DEFAULT_LOOP_COUNT;
int num_threads = DEFAULT_THREAD_COUNT;
int power       = DEFAULT_CLUSTER;

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int benchmark_graph(const char* graph_name, const char* model_file, int img_h, int img_w, int c, int n)
{
    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * c;
    int dims[] = {n, c, img_h, img_w};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

    if (input_data == NULL)
    {
        fprintf(stderr, "malloc input data buffer failed\n");
        return -1;
    }
    memset(input_data, 1, img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (prerun_graph_multithread(graph, power, num_threads) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* warning up graph */
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < loop_counts; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }

    fprintf(stderr, "%20s  min = %7.2f ms   max = %7.2f ms   avg = %7.2f ms\n", graph_name, min_time, max_time, total_time / loop_counts);

    /* release tengine graph */
    free(input_data);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    return 0;
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n  [-r loop_count] [-t thread_count] [-p cpu affinity, 0:auto, 1:big, 2:middle, 3:little] [-s net]\n");
}

int main(int argc, char* argv[])
{
    int select_num  = -1;

    int res;
    while ((res = getopt(argc, argv, "r:t:p:s:h")) != -1)
    {
        switch (res)
        {
            case 'r':
                loop_counts = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'p':
                power = atoi(optarg);
                break;
            case 's':
                select_num = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    fprintf(stderr, "loop_counts = %d\n", loop_counts);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "power       = %d\n", power);

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* run benchmarks */
    switch(select_num)
    {
        case 0:
            benchmark_graph("squeezenet_v1.1",  "./models/squeezenet_v1.1_benchmark.tmfile",    227, 227, 3, 1);
            break;
        case 1:
            benchmark_graph("mobilenetv1",      "./models/mobilenet_benchmark.tmfile",          224, 224, 3, 1);
            break;
        case 2:
            benchmark_graph("mobilenetv2",      "./models/mobilenet_v2_benchmark.tmfile",       224, 224, 3, 1);
            break;
        case 3:
            benchmark_graph("mobilenetv3",      "./models/mobilenet_v3_benchmark.tmfile",       224, 224, 3, 1);
            break;
        case 4:
            benchmark_graph("shufflenetv2",     "./models/shufflenet_v2_benchmark.tmfile",      224, 224, 3, 1);
            break;
        case 5:
            benchmark_graph("resnet18",         "./models/resnet18_benchmark.tmfile",           224, 224, 3, 1);
            break;
        case 6:
            benchmark_graph("resnet50",         "./models/resnet50_benchmark.tmfile",           224, 224, 3, 1);
            break;
        case 7:
            benchmark_graph("googlenet",        "./models/googlenet_benchmark.tmfile",          224, 224, 3, 1);
            break;
        case 8:
            benchmark_graph("inceptionv3",      "./models/inception_v3_benchmark.tmfile",       299, 299, 3, 1);
            break;
        case 9:
            benchmark_graph("vgg16",            "./models/vgg16_benchmark.tmfile",              224, 224, 3, 1);
            break;
        case 10:
            benchmark_graph("mssd",             "./models/mssd_benchmark.tmfile",               300, 300, 3, 1);
            break;
        case 11:
            benchmark_graph("retinaface",       "./models/retinaface_benchmark.tmfile",         320, 240, 3, 1);
            break;
        case 12:
            benchmark_graph("yolov3_tiny",      "./models/yolov3_tiny_benchmark.tmfile",        416, 416, 3, 1);
            break;
        case 13:
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
//            benchmark_graph("vgg16",            "./models/vgg16_benchmark.tmfile",              224, 224, 3, 1);
            benchmark_graph("mssd",             "./models/mssd_benchmark.tmfile",               300, 300, 3, 1);
            benchmark_graph("retinaface",       "./models/retinaface_benchmark.tmfile",         320, 240, 3, 1);
            benchmark_graph("yolov3_tiny",      "./models/yolov3_tiny_benchmark.tmfile",        416, 416, 3, 1);
            benchmark_graph("mobilefacenets",   "./models/mobilefacenets_benchmark.tmfile",     112, 112, 3, 1);
    }

    /* release tengine */
    release_tengine();
    fprintf(stderr, "ALL TEST DONE\n");

    return 0;
}
