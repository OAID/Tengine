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
 * Author: 774074168@qq.com
 * original model: https://github.com/PeterL1n/RobustVideoMatting
 */

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include <map>
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

const char* example_params = "example arguments:\n"
                             "\t./tm_rvm -m ./rvm_mobilenetv3.tmfile -i ./input.jpg -t 4\n"
                             "\t./tm_rvm -m ./rvm_mobilenetv3.tmfile -v ./input.mp4 -t 4\n";

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n\t[-m model_file] [-i image_file] [-v video_file] [-t thread_count]\n");
    fprintf(stderr, "%s\n", example_params);
}

void get_input_data(cv::Mat sample, float* input1_data, float* input2_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
{
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

    cv::resize(img, img, cv::Size(letterbox_cols, letterbox_rows));

    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3, cv::Scalar(0, 0, 0));

    img.convertTo(img_new, CV_32FC3);
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
                input1_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
                input2_data[out_index] = img_data[in_index] / 255.0;
            }
        }
    }
}

int video_infer(const char* video_file, const char* model_file, int num_thread)
{
    const float mean[3] = {123.68f, 116.78f, 103.94f};
    const float scale[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    // allow none square letterbox, set default letterbox size
    int letterbox_rows = 512;
    int letterbox_cols = 512;

    cv::VideoCapture cap(video_file);
    if (!cap.isOpened())
    {
        fprintf(stderr, "video %s open failed\n", video_file);
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
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = letterbox_rows * letterbox_cols * 3;
    int dims[] = {1, 3, int(letterbox_rows), int(letterbox_cols)};
    int dims1[] = {1, 16, 256, 256};
    int dims2[] = {1, 20, 128, 128};
    int dims3[] = {1, 40, 64, 64};
    int dims4[] = {1, 64, 32, 32};
    std::vector<float> input1_data(img_size);
    std::vector<float> input2_data(img_size);
    std::vector<float> input_data1(dims1[1] * dims1[2] * dims1[3], 0);
    std::vector<float> input_data2(dims2[1] * dims2[2] * dims2[3], 0);
    std::vector<float> input_data3(dims3[1] * dims3[2] * dims3[3], 0);
    std::vector<float> input_data4(dims4[1] * dims4[2] * dims4[3], 0);

    tensor_t input_tensor1 = get_graph_tensor(graph, "src1");
    tensor_t input_tensor2 = get_graph_tensor(graph, "src2");
    tensor_t r1i_tensor = get_graph_tensor(graph, "r1i");
    tensor_t r2i_tensor = get_graph_tensor(graph, "r2i");
    tensor_t r3i_tensor = get_graph_tensor(graph, "r3i");
    tensor_t r4i_tensor = get_graph_tensor(graph, "r4i");
    if (input_tensor1 == nullptr || r1i_tensor == nullptr || r2i_tensor == nullptr || r3i_tensor == nullptr || r4i_tensor == nullptr || input_tensor2 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor1, dims, 4) < 0 || set_tensor_shape(r1i_tensor, dims1, 4) < 0 || set_tensor_shape(r2i_tensor, dims2, 4) < 0 || set_tensor_shape(r3i_tensor, dims3, 4) < 0 || set_tensor_shape(r4i_tensor, dims4, 4) < 0 || set_tensor_shape(input_tensor2, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor1, input1_data.data(), img_size * 4) < 0 || set_tensor_buffer(r1i_tensor, input_data1.data(), dims1[1] * dims1[2] * dims1[3] * 4) < 0 || set_tensor_buffer(r2i_tensor, input_data2.data(), dims2[1] * dims2[2] * dims2[3] * 4) < 0 || set_tensor_buffer(r3i_tensor, input_data3.data(), dims3[1] * dims3[2] * dims3[3] * 4) < 0 || set_tensor_buffer(r4i_tensor, input_data4.data(), dims4[1] * dims4[2] * dims4[3] * 4) < 0 || set_tensor_buffer(input_tensor2, input2_data.data(), img_size * 4) < 0)
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

    tensor_t fgr;
    tensor_t pha;

    cv::Mat bgr = cv::Mat(letterbox_rows, letterbox_cols, CV_8UC3);
    bgr.setTo(cv::Scalar(155, 255, 120));

    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            continue;

        /* prepare process input data, set the data mem to input tensor */
        get_input_data(frame, input1_data.data(), input2_data.data(), letterbox_rows, letterbox_cols, mean, scale);

        /* run graph */
        double min_time = DBL_MAX;
        double max_time = DBL_MIN;
        double total_time = 0.;
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        fprintf(stderr, "thread %d, infer time %.2f ms\n", num_thread, total_time);
        fprintf(stderr, "--------------------------------------\n");

        fgr = get_graph_tensor(graph, "fgr");
        pha = get_graph_tensor(graph, "pha");

        r1i_tensor = get_graph_tensor(graph, "r1o");
        r2i_tensor = get_graph_tensor(graph, "r2o");
        r3i_tensor = get_graph_tensor(graph, "r3o");
        r4i_tensor = get_graph_tensor(graph, "r4o");

        float* fgr_data = (float*)get_tensor_buffer(fgr);
        float* pha_data = (float*)get_tensor_buffer(pha);

        cv::Mat fr = cv::Mat(letterbox_rows, letterbox_cols, CV_32FC3);
        for (int i = 0; i < letterbox_rows; i++)
        {
            for (int j = 0; j < letterbox_cols; j++)
            {
                fr.at<cv::Vec3f>(i, j)[2] = fgr_data[0 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
                fr.at<cv::Vec3f>(i, j)[1] = fgr_data[1 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
                fr.at<cv::Vec3f>(i, j)[0] = fgr_data[2 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
            }
        }

        cv::Mat fr8U;
        fr.convertTo(fr8U, CV_8UC3, 255.0, 0);

        cv::Mat alpha = cv::Mat(letterbox_rows, letterbox_cols, CV_32FC1, pha_data);
        cv::Mat alpha8U;
        alpha.convertTo(alpha8U, CV_8UC3, 255.0, 0);

        cv::Mat comp = cv::Mat(letterbox_rows, letterbox_cols, CV_8UC3);
        for (int h = 0; h < letterbox_rows; h++)
        {
            for (int w = 0; w < letterbox_cols; w++)
            {
                comp.at<cv::Vec3b>(h, w)[0] = fr8U.at<cv::Vec3b>(h, w)[0] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[0] * (1 - alpha.at<float>(h, w));
                comp.at<cv::Vec3b>(h, w)[1] = fr8U.at<cv::Vec3b>(h, w)[1] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[1] * (1 - alpha.at<float>(h, w));
                comp.at<cv::Vec3b>(h, w)[2] = fr8U.at<cv::Vec3b>(h, w)[2] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[2] * (1 - alpha.at<float>(h, w));
            }
        }

        cv::imshow("alpha", alpha8U);
        cv::imshow("fgr", fr8U);
        cv::imshow("comp", comp);
        if ('q' == cv::waitKey(1)) break;
    }

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

int image_infer(const char* image_file, const char* model_file, int num_thread)
{
    const float mean[3] = {123.68f, 116.78f, 103.94f};
    const float scale[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
    // allow none square letterbox, set default letterbox size
    int letterbox_rows = 512;
    int letterbox_cols = 512;

    cv::Mat img = cv::imread(image_file);
    if (img.empty())
    {
        fprintf(stderr, "image %s open failed\n", image_file);
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
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = letterbox_rows * letterbox_cols * 3;
    int dims[] = {1, 3, int(letterbox_rows), int(letterbox_cols)};
    int dims1[] = {1, 16, 256, 256};
    int dims2[] = {1, 20, 128, 128};
    int dims3[] = {1, 40, 64, 64};
    int dims4[] = {1, 64, 32, 32};
    std::vector<float> input1_data(img_size);
    std::vector<float> input2_data(img_size);
    std::vector<float> input_data1(dims1[1] * dims1[2] * dims1[3], 0);
    std::vector<float> input_data2(dims2[1] * dims2[2] * dims2[3], 0);
    std::vector<float> input_data3(dims3[1] * dims3[2] * dims3[3], 0);
    std::vector<float> input_data4(dims4[1] * dims4[2] * dims4[3], 0);

    tensor_t input_tensor1 = get_graph_tensor(graph, "src1");
    tensor_t input_tensor2 = get_graph_tensor(graph, "src2");
    tensor_t r1i_tensor = get_graph_tensor(graph, "r1i");
    tensor_t r2i_tensor = get_graph_tensor(graph, "r2i");
    tensor_t r3i_tensor = get_graph_tensor(graph, "r3i");
    tensor_t r4i_tensor = get_graph_tensor(graph, "r4i");
    if (input_tensor1 == nullptr || r1i_tensor == nullptr || r2i_tensor == nullptr || r3i_tensor == nullptr || r4i_tensor == nullptr || input_tensor2 == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor1, dims, 4) < 0 || set_tensor_shape(r1i_tensor, dims1, 4) < 0 || set_tensor_shape(r2i_tensor, dims2, 4) < 0 || set_tensor_shape(r3i_tensor, dims3, 4) < 0 || set_tensor_shape(r4i_tensor, dims4, 4) < 0 || set_tensor_shape(input_tensor2, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor1, input1_data.data(), img_size * 4) < 0 || set_tensor_buffer(r1i_tensor, input_data1.data(), dims1[1] * dims1[2] * dims1[3] * 4) < 0 || set_tensor_buffer(r2i_tensor, input_data2.data(), dims2[1] * dims2[2] * dims2[3] * 4) < 0 || set_tensor_buffer(r3i_tensor, input_data3.data(), dims3[1] * dims3[2] * dims3[3] * 4) < 0 || set_tensor_buffer(r4i_tensor, input_data4.data(), dims4[1] * dims4[2] * dims4[3] * 4) < 0 || set_tensor_buffer(input_tensor2, input2_data.data(), img_size * 4) < 0)
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

    tensor_t fgr;
    tensor_t pha;

    cv::Mat bgr = cv::Mat(letterbox_rows, letterbox_cols, CV_8UC3);
    bgr.setTo(cv::Scalar(155, 255, 120));

    /* prepare process input data, set the data mem to input tensor */
    get_input_data(img, input1_data.data(), input2_data.data(), letterbox_rows, letterbox_cols, mean, scale);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    double start = get_current_time();
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }
    double end = get_current_time();
    double cur = end - start;
    total_time += cur;
    fprintf(stderr, "thread %d, infer time %.2f ms\n", num_thread, total_time);
    fprintf(stderr, "--------------------------------------\n");

    fgr = get_graph_tensor(graph, "fgr");
    pha = get_graph_tensor(graph, "pha");

    r1i_tensor = get_graph_tensor(graph, "r1o");
    r2i_tensor = get_graph_tensor(graph, "r2o");
    r3i_tensor = get_graph_tensor(graph, "r3o");
    r4i_tensor = get_graph_tensor(graph, "r4o");

    float* fgr_data = (float*)get_tensor_buffer(fgr);
    float* pha_data = (float*)get_tensor_buffer(pha);

    cv::Mat fr = cv::Mat(letterbox_rows, letterbox_cols, CV_32FC3);
    for (int i = 0; i < letterbox_rows; i++)
    {
        for (int j = 0; j < letterbox_cols; j++)
        {
            fr.at<cv::Vec3f>(i, j)[2] = fgr_data[0 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
            fr.at<cv::Vec3f>(i, j)[1] = fgr_data[1 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
            fr.at<cv::Vec3f>(i, j)[0] = fgr_data[2 * letterbox_cols * letterbox_rows + i * letterbox_cols + j];
        }
    }

    cv::Mat fr8U;
    fr.convertTo(fr8U, CV_8UC3, 255.0, 0);

    cv::Mat alpha = cv::Mat(letterbox_rows, letterbox_cols, CV_32FC1, pha_data);
    cv::Mat alpha8U;
    alpha.convertTo(alpha8U, CV_8UC3, 255.0, 0);

    cv::Mat comp = cv::Mat(letterbox_rows, letterbox_cols, CV_8UC3);
    for (int h = 0; h < letterbox_rows; h++)
    {
        for (int w = 0; w < letterbox_cols; w++)
        {
            comp.at<cv::Vec3b>(h, w)[0] = fr8U.at<cv::Vec3b>(h, w)[0] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[0] * (1 - alpha.at<float>(h, w));
            comp.at<cv::Vec3b>(h, w)[1] = fr8U.at<cv::Vec3b>(h, w)[1] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[1] * (1 - alpha.at<float>(h, w));
            comp.at<cv::Vec3b>(h, w)[2] = fr8U.at<cv::Vec3b>(h, w)[2] * alpha.at<float>(h, w) + bgr.at<cv::Vec3b>(h, w)[2] * (1 - alpha.at<float>(h, w));
        }
    }

    cv::imwrite("RobustVideoMatting_alpha.jpg", alpha8U);
    cv::imwrite("RobustVideoMatting_fgr.jpg", fr8U);
    cv::imwrite("RobustVideoMatting_comp.jpg", comp);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* video_file = nullptr;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:i:v:t:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'v':
            video_file = optarg;
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
        return -1;

    if (video_file != nullptr)
    {
        if (video_infer(video_file, model_file, num_thread) < 0)
            return -1;
    }
    else if (image_file != nullptr)
    {
        if (image_infer(image_file, model_file, num_thread) < 0)
            return -1;
    }
    else
    {
        fprintf(stderr, "Error: Image file and Video file not specified!\n");
        show_usage();
        return -1;
    }

    return 0;
}
