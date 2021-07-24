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
 * original model: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
 */

#include <vector>
#include <algorithm>
#include <iostream>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define num_featuremap       4
#define hard_nms             1
#define blending_nms         2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
#define clip(x, y)           (x < 0 ? 0 : (x > y ? y : x))

typedef struct FaceInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

//input image size
const int g_tensor_in_w = 320;
const int g_tensor_in_h = 240;

const float g_score_threshold = 0.7f;
const float g_iou_threshold = 0.3f;
const float g_center_variance = 0.1f;
const float g_size_variance = 0.2f;
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
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-r repeat_count] [-t thread_count]\n\
[example]: tm_ultraface -m version-RFB-320_simplified.tmfile \n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char model_string[] = "./models/rfb-320.tmfile";
    char* model_file = model_string;

    int res;
    while ((res = getopt(argc, argv, "m:r:t:h:")) != -1)
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
    std::string model_name = "version-RFB-320_simplified";

    /* check files */
    if (model_file == NULL)
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
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (graph == NULL)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = g_tensor_in_h * g_tensor_in_w * 3;
    int dims[] = {1, 3, g_tensor_in_h, g_tensor_in_w}; // nchw
    float* input_data = (float*)malloc(img_size * sizeof(float));

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    //save input
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;

    fp = fopen(input_file.c_str(), "rb");
    if (fread(input_data, sizeof(float), img_size, fp) == 0)
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
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    tensor_t boxs_tensor = get_graph_output_tensor(graph, 0, 0);
    tensor_t scores_tensor = get_graph_output_tensor(graph, 1, 0);

    float* boxs_data = (float*)get_tensor_buffer(boxs_tensor);
    float* scores_data = (float*)get_tensor_buffer(scores_tensor);

    // save output_data
    int output_size1 = get_tensor_buffer_size(boxs_tensor) / sizeof(float);
    int output_size2 = get_tensor_buffer_size(scores_tensor) / sizeof(float);
    std::string reference_file1 = "./data/" + model_name + "_out1.bin";
    std::string reference_file2 = "./data/" + model_name + "_out2.bin";
    std::vector<float> reference_data1(output_size1);
    std::vector<float> reference_data2(output_size2);
    FILE* fp1;
    //write

    //read
    fp1 = fopen(reference_file1.c_str(), "rb");
    if (!fp || fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);

    fp1 = fopen(reference_file2.c_str(), "rb");
    if (!fp || fread(reference_data2.data(), sizeof(float), output_size2, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);
    int ret1 = float_mismatch(boxs_data, reference_data1.data(), output_size1);
    int ret2 = float_mismatch(scores_data, reference_data2.data(), output_size2);

    int ret = (ret1 | ret2);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret;
}
