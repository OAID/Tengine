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
 * Author: xwwang@openailab.com
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>

#include "common.h"
#include "tengine_c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

image rgb2gray_testcrnn(image src)
{
    image res;
    res.h = src.h;
    res.w = src.w;
    res.c = 1;
    res.data = ( float* )malloc(sizeof(float) * res.h * res.w);

    for (int i = 0; i < res.h; i++)
    {
        for (int j = 0; j < res.w; j++)
        {
            float r = src.data[0 * src.h * src.w + i * src.w + j];
            float g = src.data[1 * src.h * src.w + i * src.w + j];
            float b = src.data[2 * src.h * src.w + i * src.w + j];
            res.data[i * res.w + j] = (r * 0.114f + g * 0.587f + b * 0.299f);
        }
    }
    free_image(src);
    return res;
}

image imread_process_testcrnn(const char* filename, int img_w, int img_h, float* means, float* scale)
{
    image out = imread(filename);
    image resImg = make_image(img_w, img_h, 1);
    out = rgb2gray_testcrnn(out);
    tengine_resize_f32(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
    resImg = imread2caffe(resImg, img_w, img_h, means, scale);
    free_image(out);
    return resImg;
}

void get_input_data_test_crnn(const char* image_file, float* input_data, int img_h, int img_w, const float* mean,
                              const float* scale)
{
    float means[3] = {mean[0], mean[1], mean[2]};
    float scales[3] = {scale[0], scale[1], scale[2]};
    image img = imread_process_testcrnn(image_file, img_w, img_h, means, scales);
    memcpy(input_data, img.data, sizeof(float) * img.c * img_w * img_h);
    free_image(img);
}

std::string read_txt(const std::string& filename, int line)
{
    std::ifstream fin;
    fin.open(filename, std::ios::in);
    std::string strVec[5530];
    int i = 0;
    while (!fin.eof())
    {
        std::string inbuf;
        getline(fin, inbuf, '\n');
        strVec[i] = inbuf;
        i = i + 1;
    }
    return strVec[line - 1];
}

void process_crnn_result(const float* ocr_data, const char* label_file)
{
    int last_idx = 0;
    // read key.txt
    std::string str1;
    for (int i = 0; i < 70; i++)
    {
        const float* idx = ocr_data + i * (5530);
        int max_index = 0;
        float max_value = -__DBL_MAX__;
        for (int j = 0; j < 5530; j++)
        {
            float loc = idx[j];
            if (loc > max_value)
            {
                max_value = loc;
                max_index = j;
            }
            if (j == 5529 && max_index - 1 != -1 && max_index != last_idx && i > 0)
            {
                str1 = read_txt(label_file, max_index);
                fprintf(stderr, "%s", str1.c_str());
                str1 = ' ';
            }
        }
        last_idx = max_index;
    }
    fprintf(stderr, "\n--------------------------------------\n");
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;
    char* label_file = nullptr;
    int img_h = 32;
    int img_w = 277;
    float mean[3] = {127.5, 127.5, 127.5};
    float scale[3] = {0.007843, 0.007843, 0.007843};

    int res;
    while ((res = getopt(argc, argv, "m:i:l:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'l':
                label_file = optarg;
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

    /* check files */
    if (model_file == nullptr)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == nullptr)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (label_file == nullptr)
    {
        fprintf(stderr, "Error: Label file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file) || !check_file_exist(label_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;

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
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    int img_size = img_h * img_w * 1;
    int dims[] = {1, 1, img_h, img_w};
    float* input_data = ( float* )malloc(img_size * sizeof(float));

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
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
    get_input_data_test_crnn(image_file, input_data, img_h, img_w, mean, scale);

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
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

    /* process the crnn result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* ocr_data = ( float* )get_tensor_buffer(output_tensor);
    process_crnn_result(ocr_data, label_file);

    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
