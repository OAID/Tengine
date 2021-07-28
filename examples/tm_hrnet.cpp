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
 * Author: qtang@openailab.com
 * Author: stevenwudi@fiture.com
 *
 * original model: https://mmpose.readthedocs.io/en/latest/papers/backbones.html#div-align-center-hrnet-cvpr-2019-div
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define LETTERBOX_ROWS       256
#define LETTERBOX_COLS       256
#define MODEL_CHANNELS       3
#define HEATMAP_CHANNEL      16

typedef struct
{
    float x;
    float y;
    float score;
} ai_point_t;

struct skeleton
{
    int connection[2];
    int left_right_neutral;
};

std::vector<skeleton> pairs = {{0, 1, 0},
                               {1, 2, 0},
                               {3, 4, 1},
                               {4, 5, 1},
                               {2, 6, 0},
                               {3, 6, 1},
                               {6, 7, 2},
                               {7, 8, 2},
                               {8, 9, 2},
                               {13, 7, 1},
                               {10, 11, 0},
                               {7, 12, 0},
                               {12, 11, 0},
                               {13, 14, 1},
                               {14, 15, 1}};

typedef struct
{
    std::vector<ai_point_t> keypoints;
    int32_t img_width = 0;
    int32_t img_heigh = 0;
    uint64_t timestamp = 0;
} ai_body_parts_s;

void FindMax2D(float* buf, int width, int height, int* max_idx_width, int* max_idx_height, float* max_value, int c)
{
    float* ptr = buf;
    *max_value = -10.f;
    *max_idx_width = 0;
    *max_idx_height = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            float score = ptr[c * height * width + h * height + w];
            if (score > *max_value)
            {
                *max_value = score;
                *max_idx_height = h;
                *max_idx_width = w;
            }
        }
    }
}

void PostProcess(float* data, ai_body_parts_s& pose, int img_h, int img_w)
{
    int heatmap_width = img_w / 4;
    int heatmap_height = img_h / 4;
    int max_idx_width, max_idx_height;
    float max_score;

    ai_point_t kp;
    for (int c = 0; c < HEATMAP_CHANNEL; ++c)
    {
        FindMax2D(data, heatmap_width, heatmap_height, &max_idx_width, &max_idx_height, &max_score, c);
        kp.x = (float)max_idx_width / (float)heatmap_width;
        kp.y = (float)max_idx_height / (float)heatmap_height;
        kp.score = max_score;
        pose.keypoints.push_back(kp);

        std::cout << "x: " << pose.keypoints[c].x * 64 << ", y: " << pose.keypoints[c].y * 64 << ", score: "
                  << pose.keypoints[c].score << std::endl;
    }
}

void draw_result(cv::Mat img, ai_body_parts_s& pose)
{
    /* recover process to draw */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;

    if ((LETTERBOX_ROWS * 1.0 / img.rows) < (LETTERBOX_COLS * 1.0 / img.cols))
        scale_letterbox = LETTERBOX_ROWS * 1.0 / img.rows;
    else
        scale_letterbox = LETTERBOX_COLS * 1.0 / img.cols;

    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    int tmp_h = (LETTERBOX_ROWS - resize_rows) / 2;
    int tmp_w = (LETTERBOX_COLS - resize_cols) / 2;

    float ratio_x = (float)img.rows / resize_rows;
    float ratio_y = (float)img.cols / resize_cols;

    for (int i = 0; i < HEATMAP_CHANNEL; i++)
    {
        int x = (int)((pose.keypoints[i].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y = (int)((pose.keypoints[i].y * LETTERBOX_ROWS - tmp_h) * ratio_y);

        x = std::max(std::min(x, (img.cols - 1)), 0);
        y = std::max(std::min(y, (img.rows - 1)), 0);

        cv::circle(img, cv::Point(x, y), 4, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    cv::Scalar color;
    cv::Point pt1;
    cv::Point pt2;
    for (auto& element : pairs)
    {
        switch (element.left_right_neutral)
        {
        case 0:
            color = cv::Scalar(255, 0, 0);
            break;
        case 1:
            color = cv::Scalar(0, 0, 255);
            break;
        default:
            color = cv::Scalar(0, 255, 0);
        }

        int x1 = (int)((pose.keypoints[element.connection[0]].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y1 = (int)((pose.keypoints[element.connection[0]].y * LETTERBOX_ROWS - tmp_h) * ratio_y);
        int x2 = (int)((pose.keypoints[element.connection[1]].x * LETTERBOX_COLS - tmp_w) * ratio_x);
        int y2 = (int)((pose.keypoints[element.connection[1]].y * LETTERBOX_ROWS - tmp_h) * ratio_y);

        x1 = std::max(std::min(x1, (img.cols - 1)), 0);
        y1 = std::max(std::min(y1, (img.rows - 1)), 0);
        x2 = std::max(std::min(x2, (img.cols - 1)), 0);
        y2 = std::max(std::min(y2, (img.rows - 1)), 0);

        pt1 = cv::Point(x1, y1);
        pt2 = cv::Point(x2, y2);
        cv::line(img, pt1, pt2, color, 2);
    }
}

void get_input_fp32_data_square(const char* image_file, float* input_data, float* mean, float* scale)
{
    cv::Mat img = cv::imread(image_file);

    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    // Currenty we only support square input.
    int resize_rows;
    int resize_cols;
    if ((LETTERBOX_ROWS * 1.0 / img.rows) < (LETTERBOX_COLS * 1.0 / img.cols * 1.0))
        scale_letterbox = 1.0 * LETTERBOX_ROWS / img.rows;
    else
        scale_letterbox = 1.0 * LETTERBOX_COLS / img.cols;

    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);
    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox
    cv::Mat img_new(LETTERBOX_COLS, LETTERBOX_ROWS, CV_32FC3,
                    cv::Scalar(0.5 / scale[0] + mean[0], 0.5 / scale[1] + mean[1], 0.5 / scale[2] + mean[2]));

    int top = (LETTERBOX_ROWS - resize_rows) / 2;
    int bot = (LETTERBOX_ROWS - resize_rows + 1) / 2;
    int left = (LETTERBOX_COLS - resize_cols) / 2;
    int right = (LETTERBOX_COLS - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    //    cv::imwrite("hrnet_lb_image.jpg", img_new); // for letterbox test
    float* img_data = (float*)img_new.data;

    /* nhwc to nchw */
    for (int h = 0; h < LETTERBOX_ROWS; h++)
    {
        for (int w = 0; w < LETTERBOX_COLS; w++)
        {
            for (int c = 0; c < MODEL_CHANNELS; c++)
            {
                int in_index = h * LETTERBOX_COLS * MODEL_CHANNELS + w * MODEL_CHANNELS + c;
                int out_index = c * LETTERBOX_ROWS * LETTERBOX_COLS + h * LETTERBOX_COLS + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
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
    int img_h = LETTERBOX_COLS;
    int img_w = LETTERBOX_ROWS;
    ai_body_parts_s pose;

    float mean[3] = {123.67f, 116.28f, 103.53f};
    float scale[3] = {0.017125f, 0.017507f, 0.017429f};

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

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
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
    get_input_fp32_data_square(image_file, input_data.data(), mean, scale);

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
    fprintf(stderr, "Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
            total_time / repeat_count);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* data = (float*)(get_tensor_buffer(output_tensor));

    PostProcess(data, pose, img_h, img_w);

    /* write some visualisation  */
    cv::Mat img_out = cv::imread(image_file);
    draw_result(img_out, pose);
    cv::imwrite("hrnet_out.jpg", img_out);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
