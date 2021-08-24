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
 * original model: https://github.com/milesial/Pytorch-UNet
 */

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <limits>
#include <iostream>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H          512
#define DEFAULT_IMG_W          512
#define DEFAULT_SCALE1         (1.f / 255.f)
#define DEFAULT_SCALE2         (1.f / 255.f)
#define DEFAULT_SCALE3         (1.f / 255.f)
#define DEFAULT_MEAN1          0
#define DEFAULT_MEAN2          0
#define DEFAULT_MEAN3          0
#define DEFAULT_LOOP_COUNT     1
#define DEFAULT_THREAD_COUNT   1
#define DEFAULT_CPU_AFFINITY   255
#define DEFAULT_CONF_THRESHOLD 0.5f

/**
 * unet model are tested based on https://github.com/milesial/Pytorch-UNet
 * pretrained model can be downloaded from https://github.com/milesial/Pytorch-UNet/releases/tag/v1.0
 * tmfile can be converted using the pretrained model
 * because of the onnx->tmfile convertion problem, keep the network input size dividable by 16 (256,512) 
 */
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

int tengine_segment(const char* model_file, const char* image_file, int img_h, int img_w, const float* mean,
                    const float* scale, int loop_count, int num_thread, int affinity, float conf_thresh)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = affinity;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
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
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    std::string model_name = "unet";
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;

    fp = fopen(input_file.c_str(), "rb");
    if (!fp || fread(input_data, sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input data file failed!\n");
        return -1;
    }
    fclose(fp);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < loop_count; i++)
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
    fprintf(stderr, "\nmodel file : %s\n", model_file);
    fprintf(stderr, "img_h, img_w, scale[3], mean[3] : %d %d , %.3f %.3f %.3f, %.1f %.1f %.1f\n", img_h, img_w,
            scale[0], scale[1], scale[2], mean[0], mean[1], mean[2]);
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of classification */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = (float*)get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    std::string reference_file1 = "./data/" + model_name + "_out.bin";
    std::vector<float> reference_data1(output_size);
    FILE* fp1;
    // fp1 = fopen(reference_file1.c_str(), "wb");
    // if (!fp1 || fwrite(output_data, sizeof(float), output_size, fp1) == 0)
    // {
    //     fprintf(stderr, "read reference data file1 failed!\n");
    //     return -1;
    // }
    // fclose(fp1);

    fp1 = fopen(reference_file1.c_str(), "rb");
    if (!fp1 || fread(reference_data1.data(), sizeof(float), output_size, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);
    int ret1 = float_mismatch(output_data, reference_data1.data(), output_size);
    /* single class segmentation */
    /* multi-class segmentation */
    /* visualization */

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret1;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] \n  [-r loop_count] [-t thread_count] [-a cpu_affinity] [-c conf_thresh]\n");
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    int cpu_affinity = DEFAULT_CPU_AFFINITY;
    float conf_thresh = DEFAULT_CONF_THRESHOLD;
    char model_string[] = "./models/unet.tmfile";
    char* model_file = model_string;
    char* image_file = NULL;
    float img_hw[2] = {0.f};
    int img_h = 0;
    int img_w = 0;
    float mean[3] = {0.f, 0.f, 0.f};
    float scale[3] = {0.f, 0.f, 0.f};

    int res;
    while ((res = getopt(argc, argv, "m:r:t:a:c:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'r':
            loop_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'a':
            cpu_affinity = atoi(optarg);
            break;
        case 'c':
            conf_thresh = atof(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

    img_h = DEFAULT_IMG_H;

    img_w = DEFAULT_IMG_W;

    scale[0] = DEFAULT_SCALE1;
    scale[1] = DEFAULT_SCALE2;
    scale[2] = DEFAULT_SCALE3;

    mean[0] = DEFAULT_MEAN1;
    mean[1] = DEFAULT_MEAN2;
    mean[2] = DEFAULT_MEAN3;

    if (tengine_segment(model_file, image_file, img_h, img_w, mean, scale, loop_count, num_thread, cpu_affinity, conf_thresh) < 0)
        return -1;

    return 0;
}
