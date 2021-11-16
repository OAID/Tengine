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
 * Author: 774074168@qq.com
 */

#include <iostream>
#include <functional>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

static void get_input_fp32_data(const char* image_file, float* input_data,
                                int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));

    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv

    cv::Mat img_new(letterbox_rows, letterbox_cols, CV_32FC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * letterbox_cols * 3 + w * 3 + c;
                int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
}
void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}
static void draw_result(const cv::Mat& bgr, std::vector<cv::Point2f> pts)
{
    cv::Scalar color1(10, 215, 255);
    cv::Scalar color2(255, 115, 55);
    cv::Scalar color3(5, 255, 55);
    cv::Scalar color4(25, 15, 255);
    cv::Scalar color5(225, 15, 55);
    for (size_t j = 0; j < 21; j++)
    {
        cv::circle(bgr, pts[j], 4, cv::Scalar(255, 0, 255), -1);
        if (j < 4)
        {
            cv::line(bgr, pts[j], pts[j + 1], color1, 2, 8);
        }
        if (j < 8 && j > 4)
        {
            cv::line(bgr, pts[j], pts[j + 1], color2, 2, 8);
        }
        if (j < 12 && j > 8)
        {
            cv::line(bgr, pts[j], pts[j + 1], color3, 2, 8);
        }
        if (j < 16 && j > 12)
        {
            cv::line(bgr, pts[j], pts[j + 1], color4, 2, 8);
        }
        if (j < 20 && j > 16)
        {
            cv::line(bgr, pts[j], pts[j + 1], color5, 2, 8);
        }
    }
    cv::line(bgr, pts[0], pts[5], color2, 2, 8);
    cv::line(bgr, pts[0], pts[9], color3, 2, 8);
    cv::line(bgr, pts[0], pts[13], color4, 2, 8);
    cv::line(bgr, pts[0], pts[17], color5, 2, 8);

    cv::imwrite("handpose_result.jpg", bgr);
}
int main(int argc, char* argv[])
{
    int repeat_count = 1;
    int num_thread = 1;
    char* model_file = nullptr;
    char* image_file = nullptr;

    int letterbox_rows = 224;
    int letterbox_cols = 224;
    int img_c = 3;

    float mean[3] = {0.f, 0.f, 0.f};
    float scale[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

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

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;
    cv::Mat img = cv::imread(image_file, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }
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
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = letterbox_rows * letterbox_cols * img_c;
    int dims[] = {1, 3, letterbox_rows, letterbox_cols};
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
    get_input_fp32_data(image_file, input_data.data(), letterbox_rows, letterbox_cols, mean, scale);

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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* get output tensor */
    tensor_t score = get_graph_tensor(graph, "score");
    tensor_t points = get_graph_tensor(graph, "points");
    float* score_data = (float*)get_tensor_buffer(score);
    float* points_data = (float*)get_tensor_buffer(points);

    std::vector<cv::Point2f> pts;
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / img.rows) < (letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)img.rows / resize_rows;
    float ratio_y = (float)img.cols / resize_cols;

    for (int i = 0; i < 21; i++)
    {
        float x = (points_data[3 * i] - tmp_w) * ratio_x;
        float y = (points_data[3 * i + 1] - tmp_h) * ratio_y;
        pts.push_back(cv::Point2f(x, y));
    }
    if (score_data[0] > 0.5)
    {
        fprintf(stderr, "Right hand\n");
    }
    else
    {
        fprintf(stderr, "Left hand\n");
    }

    draw_result(img, pts);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
