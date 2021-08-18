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
 * original model: https://github.com/ultralytics/yolov5
 */

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

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

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    char model_string[] = "./models/yolov5s.tmfile";
    char* model_file = model_string;

    int img_c = 3;

    // allow none square letterbox, set default letterbox size
    int letterbox_rows = 640;
    int letterbox_cols = 640;

    int repeat_count = 1;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:r:t:h:")) != -1)
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

    int img_size = letterbox_rows * letterbox_cols * img_c;
    int dims[] = {1, 12, int(letterbox_rows / 2), int(letterbox_cols / 2)};
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
    std::string model_name = "yolov5s";
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (fread(input_data.data(), sizeof(float), img_size, fp) == 0)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* yolov5 postprocess */
    // 0: 1, 3, 20, 20, 85
    // 1: 1, 3, 40, 40, 85
    // 2: 1, 3, 80, 80, 85
    tensor_t p8_output = get_graph_output_tensor(graph, 0, 0);
    tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
    tensor_t p32_output = get_graph_output_tensor(graph, 2, 0);

    float* p8_data = (float*)get_tensor_buffer(p8_output);
    float* p16_data = (float*)get_tensor_buffer(p16_output);
    float* p32_data = (float*)get_tensor_buffer(p32_output);

    /* postprocess */
    int output_size1 = get_tensor_buffer_size(p8_output) / sizeof(float);
    int output_size2 = get_tensor_buffer_size(p16_output) / sizeof(float);
    int output_size3 = get_tensor_buffer_size(p32_output) / sizeof(float);
    std::string reference_file1 = "./data/" + model_name + "_out1.bin";
    std::string reference_file2 = "./data/" + model_name + "_out2.bin";
    std::string reference_file3 = "./data/" + model_name + "_out3.bin";
    std::vector<float> reference_data1(output_size1), reference_data2(output_size2), reference_data3(output_size3);
    FILE* fp1;
    fp1 = fopen(reference_file1.c_str(), "rb");
    if (fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);
    fp1 = fopen(reference_file2.c_str(), "rb");
    if (fread(reference_data2.data(), sizeof(float), output_size2, fp1) == 0)
    {
        fprintf(stderr, "read reference data file2 failed!\n");
        return -1;
    }
    fclose(fp1);
    fp1 = fopen(reference_file3.c_str(), "rb");
    if (fread(reference_data3.data(), sizeof(float), output_size3, fp1) == 0)
    {
        fprintf(stderr, "read reference data file3 failed!\n");
        return -1;
    }
    fclose(fp1);
    int ret1 = float_mismatch(p8_data, reference_data1.data(), output_size1);
    int ret2 = float_mismatch(p16_data, reference_data2.data(), output_size2);
    int ret3 = float_mismatch(p32_data, reference_data3.data(), output_size3);

    int ret = (ret1 | ret2 | ret3);
    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return ret;
}
