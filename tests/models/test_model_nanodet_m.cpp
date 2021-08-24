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
 * original model: https://github.com/RangiLyu/nanodet
 */

/* comment: do post softmax within model graph(not recommanded) */
#define TRY_POST_SOFTMAX

/* std c++ includes */
#include <vector>
#include <algorithm>
#include <iostream>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

// tengine output tensor names
const char* cls_pred_name[] = {
    "cls_pred_stride_8", "cls_pred_stride_16", "cls_pred_stride_32"};
const char* dis_pred_name[] = {
#ifdef TRY_POST_SOFTMAX
    "dis_pred_stride_8", "dis_pred_stride_16", "dis_pred_stride_32"
#else  /* !TRY_POST_SOFTMAX */
    "dis_sm_stride_8", "dis_sm_stride_16", "dis_sm_stride_32"
#endif /* TRY_POST_SOFTMAX */
};
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

static void show_usage()
{
    fprintf(stderr, "[Usage]: [-h]\n");
    fprintf(stderr, "   [-m model_file] [-r repeat_count] [-t thread_count] [-o output_file]\n");
}

int main(int argc, char* argv[])
{
    const char* model_file = "./models/nanodet.tmfile";
    const float mean[3] = {103.53f, 116.28f, 123.675f}; // bgr
    const float norm[3] = {0.017429f, 0.017507f, 0.017125f};

    int repeat_count = 1;
    int num_thread = 1;

    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    int res;
    while ((res = getopt(argc, argv, "m:i:o:r:t:h:")) != -1)
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
    {
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (0 != init_tengine())
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    int img_size = 320 * 320 * 3; // lb.w * lb.h * lb.c;

    std::string model_name = "nanodet";
    std::string input_file = "./data/" + model_name + "_in.bin";
    std::vector<float> input_data(img_size * sizeof(float));

    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (fread(input_data.data(), sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input data file failed!\n");
        return -1;
    }
    fclose(fp);
    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }
    /* prerun graph to infer shape, and set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

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

    int ret = 0;
    /* nanodet_m postprocess */
    // std::vector<Object> proposals, objects;
    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        tensor_t cls_tensor = get_graph_tensor(graph, cls_pred_name[stride_index]);
        tensor_t dis_tensor = get_graph_tensor(graph, dis_pred_name[stride_index]);
        if (NULL == cls_tensor || NULL == dis_tensor)
        {
            fprintf(stderr, "get graph tensor failed\n");
            return -1;
        }
        float* cls_pred = (float*)get_tensor_buffer(cls_tensor);
        float* dis_pred = (float*)get_tensor_buffer(dis_tensor);

        // save output_data
        int output_size1 = get_tensor_buffer_size(cls_tensor) / sizeof(float);
        int output_size2 = get_tensor_buffer_size(dis_tensor) / sizeof(float);
        std::string reference_file1 = "./data/" + model_name + "_out" + std::to_string(stride_index * 2 + 1) + ".bin";
        std::string reference_file2 = "./data/" + model_name + "_out" + std::to_string(stride_index * 2 + 2) + ".bin";
        std::vector<float> reference_data1(output_size1);
        std::vector<float> reference_data2(output_size2);
        FILE* fp1;
        //read
        fp1 = fopen(reference_file1.c_str(), "rb");
        if (!fp1 || fread(reference_data1.data(), sizeof(float), output_size1, fp1) == 0)
        {
            fprintf(stderr, "read reference data file1 failed!\n");
            return -1;
        }
        fclose(fp1);
        fp1 = fopen(reference_file2.c_str(), "rb");
        if (!fp1 || fread(reference_data2.data(), sizeof(float), output_size2, fp1) == 0)
        {
            fprintf(stderr, "read reference data file2 failed!\n");
            return -1;
        }
        fclose(fp1);
        int ret1 = float_mismatch(cls_pred, reference_data1.data(), output_size1);
        int ret2 = float_mismatch(dis_pred, reference_data2.data(), output_size2);
        ret = ret | (ret1 | ret2);
    }

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return ret;
}
