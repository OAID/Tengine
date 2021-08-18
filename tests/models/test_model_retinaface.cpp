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
 * original model: https://github.com/deepinsight/insightface/tree/master/RetinaFace#retinaface-pretrained-models
 */

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/examples/retinaface.cpp
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

#include <vector>
#include <string>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common.h"

#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define MODEL_PATH "models/retinaface.tmfile"

const float CONF_THRESH = 0.8f;
const float NMS_THRESH = 0.4f;

const char* input_name = "data";

const char* bbox_name[3] = {"face_rpn_bbox_pred_stride32", "face_rpn_bbox_pred_stride16", "face_rpn_bbox_pred_stride8"};
const char* score_name[3] = {"face_rpn_cls_prob_reshape_stride32", "face_rpn_cls_prob_reshape_stride16",
                             "face_rpn_cls_prob_reshape_stride8"};
const char* landmark_name[3] = {"face_rpn_landmark_pred_stride32", "face_rpn_landmark_pred_stride16",
                                "face_rpn_landmark_pred_stride8"};

const int stride[3] = {32, 16, 8};

const float scales[3][2] = {{32.f, 16.f}, {8.f, 4.f}, {2.f, 1.f}};

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
    printf("[Usage]:  [-h]\n    [-m model_file]  [-r repeat_count] [-t thread_count] [-n device_name]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;

    char model_string[] = MODEL_PATH;
    const char* model_file = model_string;
    const char* device_name = "";

    int res;
    while ((res = getopt(argc, argv, "m:r:t:h:n:")) != -1)
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
        case 'n':
            device_name = optarg;
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        printf("Error: Tengine model file not specified!\n");
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
    int ret = init_tengine();
    if (0 != ret)
    {
        printf("Init tengine-lite failed.\n");
        return -1;
    }

    printf("tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (graph == nullptr)
    {
        printf("Load model to graph failed.\n");
        return -1;
    }

    /* prepare process input data */
    int height = 1150;
    int width = 2048;
    int img_size = height * width * 3;
    std::vector<float> image_data(img_size * sizeof(float));

    std::string model_name = "retinaface";
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "open input file %s failed!\n", input_file.c_str());
        return -1;
    }
    if (!fp || fread(image_data.data(), sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input file %s failed!\n", input_file.c_str());
        return -1;
    }
    fclose(fp);

    /* set the input shape to initial the graph, and pre-run graph to infer shape */

    int dims[] = {1, 3, height, width};

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    if (nullptr == input_tensor)
    {
        printf("Get input tensor failed\n");
        return -1;
    }

    if (0 != set_tensor_shape(input_tensor, dims, 4))
    {
        printf("Set input tensor shape failed\n");
        return -1;
    }

    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, image_data.data(), img_size * sizeof(float)) < 0)
    {
        printf("Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (0 != prerun_graph_multithread(graph, opt))
    {
        printf("Pre-run graph failed\n");
        return -1;
    }

    /* run graph */
    float min_time = FLT_MAX, max_time = 0, total_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            printf("Run graph failed\n");
            return -1;
        }
        double end = get_current_time();

        float cur = float(end - start);

        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    printf("Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
           num_thread, total_time / (float)repeat_count, max_time, min_time);
    printf("--------------------------------------\n");

    /* process the detection result */
    ret = 0;
    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        // ==================================================================
        // ========== This part is to get tensor information ================
        // ==================================================================
        tensor_t score_blob_tensor = get_graph_tensor(graph, score_name[stride_index]);
        tensor_t bbox_blob_tensor = get_graph_tensor(graph, bbox_name[stride_index]);
        tensor_t landmark_blob_tensor = get_graph_tensor(graph, landmark_name[stride_index]);

        int score_blob_dims[MAX_SHAPE_DIM_NUM] = {0};
        int bbox_blob_dims[MAX_SHAPE_DIM_NUM] = {0};
        int landmark_blob_dims[MAX_SHAPE_DIM_NUM] = {0};

        get_tensor_shape(score_blob_tensor, score_blob_dims, MAX_SHAPE_DIM_NUM);
        get_tensor_shape(bbox_blob_tensor, bbox_blob_dims, MAX_SHAPE_DIM_NUM);
        get_tensor_shape(landmark_blob_tensor, landmark_blob_dims, MAX_SHAPE_DIM_NUM);

        float* score_blob = (float*)get_tensor_buffer(score_blob_tensor);
        float* bbox_blob = (float*)get_tensor_buffer(bbox_blob_tensor);
        float* landmark_blob = (float*)get_tensor_buffer(landmark_blob_tensor);

        // save output_data
        int output_size1 = get_tensor_buffer_size(score_blob_tensor) / sizeof(float);
        int output_size2 = get_tensor_buffer_size(bbox_blob_tensor) / sizeof(float);
        int output_size3 = get_tensor_buffer_size(landmark_blob_tensor) / sizeof(float);
        std::string reference_file1 = "./data/" + model_name + "_out" + std::to_string(stride_index * 3 + 1) + ".bin";
        std::string reference_file2 = "./data/" + model_name + "_out" + std::to_string(stride_index * 3 + 2) + ".bin";
        std::string reference_file3 = "./data/" + model_name + "_out" + std::to_string(stride_index * 3 + 3) + ".bin";
        std::vector<float> reference_data1(output_size1);
        std::vector<float> reference_data2(output_size2);
        std::vector<float> reference_data3(output_size3);
        FILE* fp1;

        //read
        fp1 = fopen(reference_file1.c_str(), "rb");
        if (fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
        {
            fprintf(stderr, "read reference %s failed!\n", reference_file1.c_str());
            return -1;
        }
        fclose(fp1);
        fp1 = fopen(reference_file2.c_str(), "rb");
        if (fread(reference_data2.data(), sizeof(float), output_size2, fp1) == 0)
        {
            fprintf(stderr, "read reference %s failed!\n", reference_file2.c_str());
            return -1;
        }
        fclose(fp1);
        fp1 = fopen(reference_file3.c_str(), "rb");
        if (fread(reference_data3.data(), sizeof(float), output_size3, fp1) == 0)
        {
            fprintf(stderr, "read reference %s failed!\n", reference_file3.c_str());
            return -1;
        }
        fclose(fp1);

        int ret1 = float_mismatch(score_blob, reference_data1.data(), output_size1);
        int ret2 = float_mismatch(bbox_blob, reference_data2.data(), output_size2);
        int ret3 = float_mismatch(landmark_blob, reference_data3.data(), output_size3);
        ret = ret | (ret1 | ret2 | ret3);
    }

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret;
}
