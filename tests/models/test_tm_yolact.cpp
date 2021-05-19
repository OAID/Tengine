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

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/examples/yolact.cpp
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "common.hpp"
#include <sys/time.h>
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define DEF_MODEL "models/yolact.tmfile"
#define DEF_IMAGE "images/ssd_car.jpg"


static int detect_yolact(const char* model_file, int repeat_count,
                         int num_thread)
{
    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    const int target_size = 550;

    int img_w = 500;
    int img_h = 375;

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = target_size * target_size * 3;
    int dims[] = {1, 3, target_size, target_size};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

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

/* prerun the graph */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    if(std::getenv("NumThreadLite"))
        opt.num_thread = atoi(std::getenv("NumThreadLite"));
    if(std::getenv("NumClusterLite"))
        opt.cluster = atoi(std::getenv("NumClusterLite"));
    if(std::getenv("DataPrecision"))
        opt.precision = atoi(std::getenv("DataPrecision"));
    if(std::getenv("REPEAT"))
        repeat_count = atoi(std::getenv("REPEAT"));
    
    std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
    std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
    std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";
    std::cout<<"Number Repeat  : [" << repeat_count <<"], use export REPEAT=10/100/1000 set\n";

    if(prerun_graph_multithread(graph, opt) < 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    // read input data
    FILE *infp;
    infp = fopen("./data/yolact_in.bin", "rb");
    if(fread(input_data, sizeof(float), img_size, infp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(infp);

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* run graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for (int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, avg_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of classification */
    tensor_t maskmaps_tensor = get_graph_output_tensor(graph, 1, 0);
    tensor_t location_tensor = get_graph_output_tensor(graph, 2, 0);
    tensor_t mask_tensor = get_graph_output_tensor(graph, 3, 0);
    tensor_t confidence_tensor = get_graph_output_tensor(graph, 4, 0);
    float* maskmaps = ( float* )get_tensor_buffer(maskmaps_tensor);
    float* location = ( float* )get_tensor_buffer(location_tensor);
    float* mask = ( float* )get_tensor_buffer(mask_tensor);
    float* confidence = ( float* )get_tensor_buffer(confidence_tensor);

    // test output data
    int maskmaps_size = get_tensor_buffer_size(maskmaps_tensor) / sizeof(float);
    int location_size = get_tensor_buffer_size(location_tensor) / sizeof(float);
    int mask_size = get_tensor_buffer_size(mask_tensor) / sizeof(float);
    int confidence_size = get_tensor_buffer_size(confidence_tensor) / sizeof(float);
    
    float* maskmaps_ref = (float*)malloc(maskmaps_size * sizeof(float));
    float* location_ref = (float*)malloc(maskmaps_size * sizeof(float));
    float* mask_ref = (float*)malloc(mask_size * sizeof(float));
    float* confidence_ref = (float*)malloc(confidence_size * sizeof(float));

    FILE *fp;
    fp=fopen("./data/yolact_maskmaps_out.bin","rb");
    if(fread(maskmaps_ref, sizeof(float), maskmaps_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return -1;
    }
    fclose(fp);

    fp=fopen("./data/yolact_location_out.bin","rb");
    if(fread(location_ref, sizeof(float), location_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return -1;
    }
    fclose(fp);

    fp=fopen("./data/yolact_mask_out.bin","rb");
    if(fread(mask_ref, sizeof(float), mask_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return -1;
    }
    fclose(fp);

    fp=fopen("./data/yolact_confidence_out.bin","rb");
    if(fread(confidence_ref, sizeof(float), confidence_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return -1;
    }
    fclose(fp);

    if(float_mismatch(maskmaps_ref, maskmaps, maskmaps_size) != true)
        return -1;
    if(float_mismatch(location_ref, location, location_size) != true)
        return -1;
    if(float_mismatch(mask_ref, mask, mask_size) != true)
        return -1;
    if(float_mismatch(confidence_ref, confidence, confidence_size) != true)
        return -1;
    free(maskmaps_ref);
    free(location_ref);
    free(mask_ref);
    free(confidence_ref);
    // test output data done

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char** argv)
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    const std::string root_path = get_root_path();
    std::string model_file;
    std::string image_file;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
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

    // load model
    if(model_file.empty())
    {
	model_file = root_path + DEF_MODEL;
	std::cout << "model file not specified,using " << model_file << " by default\n";
    }
    if(image_file.empty())
    {
        image_file = root_path + DEF_IMAGE;
	std::cout << "image file not specified,using " << image_file << " by default\n";
    }
    // check file
    if((!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
	return -1;
    }

    detect_yolact(model_file.c_str(), repeat_count, num_thread);

    return 0;
}
