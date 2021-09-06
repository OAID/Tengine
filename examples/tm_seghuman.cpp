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
 * software distributed under the License is distributed oan an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: hbshi@openailab.com
 */

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

void get_input_fp32_data(const char* image_file, float* input_data, int inputH, int inputW)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;
    cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(inputW, inputH));

    cv::Mat img_new;
    img.convertTo(img_new, CV_32FC3);
    auto* img_data = (float*)img_new.data;

    for (int h = 0; h < inputH; h++)
    {
        for (int w = 0; w < inputW; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * inputW * 3 + w * 3 + c;
                int out_index = c * inputW * inputH + h * inputW + w;
                float tmp = (img_data[in_index] / 255.f - 0.5f) / 0.5f;
                input_data[out_index] = tmp;
            }
        }
    }
}

static void draw_human_seg_result(const cv::Mat& bgr, const float* data, int target_size_h, int target_size_w)
{
    const int palette[][3] = {{66, 128, 84},
                              {70, 89, 156}};
    cv::Mat image = bgr.clone();

    /* get class index */
    cv::Mat segidx = cv::Mat(target_size_h, target_size_w, CV_8UC1);
    int index = 0;
    for (int i = 0; i < target_size_h; ++i)
    {
        for (int j = 0; j < target_size_w; ++j)
        {
            float tmp0 = data[i * target_size_w + j];
            float tmp1 = data[target_size_w * target_size_h + i * target_size_w + j];
            if (tmp0 < tmp1)
            {
                segidx.data[index] = 1;
            }
            else
            {
                segidx.data[index] = 0;
            }
            index++;
        }
    }

    cv::Mat maskResize;
    cv::resize(segidx, maskResize, cv::Size(image.cols, image.rows), 0, 0, cv::INTER_LINEAR);

    for (int h = 0; h < image.rows; h++)
    {
        auto* pRgb = image.ptr<cv::Vec3b>(h);
        for (int w = 0; w < image.cols; w++)
        {
            int colorIndex = *maskResize.ptr<uint8_t>(h, w);
            if (colorIndex == 1)
            {
                pRgb[w] = cv::Vec3b(palette[colorIndex][2] * 0.6 + pRgb[w][2] * 0.4,
                                    palette[colorIndex][1] * 0.6 + pRgb[w][1] * 0.4,
                                    palette[colorIndex][0] * 0.6 + pRgb[w][0] * 0.4);
            }
            else
            {
                pRgb[w] = cv::Vec3b(palette[colorIndex][2],
                                    palette[colorIndex][1],
                                    palette[colorIndex][0]);
            }
        }
    }
    cv::imwrite("seg_human_result.jpg", image);
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
    int img_h = 224;
    int img_w = 398;

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
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default: break;
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

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = 2;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    auto* input_data = (float*)malloc(img_size * sizeof(float));

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
    get_input_fp32_data(image_file, input_data, img_h, img_w);

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
    printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
           total_time / repeat_count);

    /* get output tensor */
    tensor_t outputTensor0 = get_graph_output_tensor(graph, 0, 0); //age_smile
    auto outputData = (float*)get_tensor_buffer(outputTensor0);
    /* post process */
    cv::Mat input = cv::imread(image_file, 1);
    draw_human_seg_result(input, outputData, img_h, img_w);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
