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
 * original model: https://github.com/MVIG-SJTU/AlphaPose
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H        320
#define DEFAULT_IMG_W        256
#define DEFAULT_SCALE1       (0.0039216)
#define DEFAULT_SCALE2       (0.0039215)
#define DEFAULT_SCALE3       (0.0039215)
#define DEFAULT_MEAN1        0.406
#define DEFAULT_MEAN2        0.457
#define DEFAULT_MEAN3        0.480
#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

const float s_keypoint_thresh = 0.2;
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
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-r repeat_count] [-t thread_count]\n");
}

bool tengine_predict(float* input_data, graph_t graph, const int input_dims[4], const int& num_thread, const int& loop_count)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return false;
    }

    if (set_tensor_shape(input_tensor, input_dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return false;
    }

    size_t input_data_size = (unsigned long)input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * sizeof(float);
    if (set_tensor_buffer(input_tensor, input_data, input_data_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return false;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return false;
    }

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < loop_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return false;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
            loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");
    return true;
}

int main(int argc, char* argv[])
{
    const char* model_file = "./models/alphapose.tmfile";
    const char* image_file = nullptr;
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check options */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

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

    int img_height = 640;
    int img_width = 640;
    int input_dims[] = {1, 3, DEFAULT_IMG_H, DEFAULT_IMG_W}; // nchw

    //read input data
    int img_size = img_height * img_width * 3;
    std::vector<float> input_data1(img_size);

    std::string model_name = "alphapose";
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (fread(input_data1.data(), sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input data file failed!\n");
        return -1;
    }
    fclose(fp);

    // run prediction
    if (false == tengine_predict(input_data1.data(), graph, input_dims, num_thread, repeat_count))
    {
        fprintf(stderr, "Run model file: %s failed.\n", model_file);
        return -1;
    }

    //post process
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    int heatmap_dims[MAX_SHAPE_DIM_NUM] = {0};
    get_tensor_shape(output_tensor, heatmap_dims, MAX_SHAPE_DIM_NUM);

    float* data = (float*)(get_tensor_buffer(output_tensor));
    int output_size1 = get_tensor_buffer_size(output_tensor) / (sizeof(float));
    std::string reference_file1 = "./data/" + model_name + "_out.bin";
    std::vector<float> reference_data1(output_size1);
    FILE* fp1;
    fp1 = fopen(reference_file1.c_str(), "rb");
    if (fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);
    int ret1 = float_mismatch(data, reference_data1.data(), output_size1);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret1;
}
