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
 * Author: sqfu@openailab.com
 * 
 * original model: https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco
 */

#include <iostream>
#include <iomanip>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
int float_mismatch(float* current, float* reference, int size)
{
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        if (fabs(tmp) > 0.0001)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            return -1;
        }
    }
    fprintf(stderr, "test pass\n");
    return 0;
}
enum
{
    YOLOV3 = 0,
    YOLO_FASTEST = 1,
    YOLO_FASTEST_XL = 2
};

using namespace std;

static void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-r repeat_count] [-t thread_count]\n");
}
struct TMat
{
    operator const float*() const
    {
        return (const float*)data;
    }

    float* row(int row) const
    {
        return (float*)data + w * row;
    }

    TMat channel_range(int start, int chn_num) const
    {
        TMat mat = {0};

        mat.batch = 1;
        mat.c = chn_num;
        mat.h = h;
        mat.w = w;
        mat.data = (float*)data + start * h * w;

        return mat;
    }

    TMat channel(int channel) const
    {
        return channel_range(channel, 1);
    }

    int batch, c, h, w;
    void* data;
};

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char model_string[] = "./models/yolo-fastest-1.1.tmfile";
    char* model_file = model_string;

    int net_w = 320;
    int net_h = 320;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'r':
            repeat_count = std::strtoul(optarg, nullptr, 10);
            break;
        case 't':
            num_thread = std::strtoul(optarg, nullptr, 10);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = net_h * net_w * 3;
    int dims[] = {1, 3, net_h, net_w}; // nchw

    std::vector<float> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    std::string model_name = "yolo-fastest-1.1";
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (!fp || fread(input_data.data(), sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input data file failed!\n");
        return -1;
    }
    fclose(fp);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
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
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */

    int output_node_num = get_graph_output_node_number(graph);
    int ret1 = 0;
    tensor_t out_tensor;
    for (int i = 0; i < output_node_num; ++i)
    {
        out_tensor = get_graph_output_tensor(graph, i, 0); //"detection_out"
        // save output_data
        std::string model_name = "yolo-fastest-1.1";
        int output_size1 = get_tensor_buffer_size(out_tensor) / sizeof(float);
        ;
        float* yolo_outputs = (float*)get_tensor_buffer(out_tensor);
        std::string reference_file1 = "./data/" + model_name + "_out" + std::to_string(i + 1) + ".bin";
        std::vector<float> reference_data1(output_size1);
        FILE* fp1;
        //read
        fp1 = fopen(reference_file1.c_str(), "rb");
        if (fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
        {
            fprintf(stderr, "read reference data file1 failed!\n");
            return -1;
        }
        fclose(fp1);
        ret1 |= float_mismatch(yolo_outputs, reference_data1.data(), output_size1);
    }
    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret1;
}
