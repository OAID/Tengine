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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <fstream>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

using namespace std;

#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE1 1.f
#define DEFAULT_SCALE2 1.f
#define DEFAULT_SCALE3 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_LOOP_COUNT 1
#define DEFAULT_THREAD_COUNT 1

void get_input_data_cv(const cv::Mat& sample, float* input_data, int img_h, int img_w, const float* mean,
                       const float* scale, int swapRB = 0)
{
    cv::Mat img;
    if (sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else if (sample.channels() == 3 && swapRB == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else
    {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                img_data++;
            }
        }
    }
}

static void sort_cls_score(cls_score* array, int left, int right)
{
    int i = left;
    int j = right;
    cls_score key;

    if (left >= right)
        return;

    memmove(&key, &array[left], sizeof(cls_score));
    while (left < right)
    {
        while (left < right && key.score >= array[right].score)
        {
            --right;
        }
        memmove(&array[left], &array[right], sizeof(cls_score));
        while (left < right && key.score <= array[left].score)
        {
            ++left;
        }
        memmove(&array[right], &array[left], sizeof(cls_score));
    }

    memmove(&array[left], &key, sizeof(cls_score));

    sort_cls_score(array, i, left - 1);
    sort_cls_score(array, left + 1, j);
}

void accuracy_top5(float* data, int class_nums, int gt_id, int& total_num, int& top1_num, int& top5_num)
{
    cls_score* cls_scores = ( cls_score* )malloc(class_nums * sizeof(cls_score));
    for (int i = 0; i < class_nums; i++)
    {
        cls_scores[i].id = i;
        cls_scores[i].score = data[i];
    }

    sort_cls_score(cls_scores, 0, class_nums - 1);
    for (int i = 0; i < 5; i++)
    {
        if (cls_scores[i].id == gt_id)
        {
            top5_num++;
            if (i == 0)
                top1_num++;
        }
        fprintf(stderr, "%f, %d\n", cls_scores[i].score, cls_scores[i].id);
    }

    total_num++;
    float acc_top1 = ((float )top1_num / (float )total_num);
    float acc_top5 = ((float )top5_num / (float )total_num);

    fprintf(stderr, "gt_id %5d, total %5d, top1 %5d(%4.2f %%), top5 %5d(%4.2f %%)\n", gt_id, total_num, top1_num, acc_top1*100, top5_num, acc_top5*100);

    free(cls_scores);
}

int tengine_classify_accurary(const char* model_file, const char* image_dir, const char* val_file, int img_h, int img_w, const float* mean,
                     const float* scale, int swapRB, int loop_count, int num_thread)
{
    int total_num = 0;
    int top1_num = 0;
    int top5_num = 0;

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
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};    // nchw
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

    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* process image files of dir */
    string val_f = val_file;
    ifstream f(val_f);
    if (!f.is_open())
    {
        printf("open %s file failed\n", val_file);
        return -1;
    }
    string line_str;
    while(getline(f, line_str))
    {
        /* get image file */
        stringstream ss(line_str);
        string image_name;
        string gt_id;
        getline(ss, image_name, ' ');
        getline(ss, gt_id, ' ');

        string image_file = image_dir + image_name;
        ifstream tmp(image_file);
        if (!tmp.is_open())
            continue;

        cv::Mat m = cv::imread(image_file, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", image_file.c_str());
            return -1;
        }

        /* process input date of image */
        get_input_data_cv(m, input_data, img_h, img_w, mean, scale, swapRB);

        /* run graph */
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        fprintf(stderr, "forward time %5.1f ms, %s\n", cur, image_file.c_str());

        /* get the result of classification */
        tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
        float* output_data = ( float* )get_tensor_buffer(output_tensor);
        int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

        accuracy_top5(output_data, output_size, stoi(gt_id), total_num, top1_num, top5_num);
        fprintf(stderr, "--------------------------------------\n");
    }

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_dir] [-v val_file]\n [-g img_h,img_w] [-s scale[0],scale[1],scale[2]] [-w "
        "mean[0],mean[1],mean[2]] [-c swapRB] [-r loop_count] [-t thread_count]\n");
    fprintf(
        stderr,
        "\nmobilenet example: \n    ./classification -m /path/to/mobilenet.tmfile -i /path/to/imagenet2012_val -v /path/to/val.txt -g 224,224 -s "
        "0.017,0.017,0.017 -w 104.007,116.669,122.679\n");
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    int swapRB = 0;
    char* model_file = nullptr;
    char* image_dir = nullptr;
    char* val_file = nullptr;
    float img_hw[2] = {0.f};
    int img_h = 0;
    int img_w = 0;
    float mean[3] = {-1.f, -1.f, -1.f};
    float scale[3] = {0.f, 0.f, 0.f};

    int res;
    while ((res = getopt(argc, argv, "m:i:v:g:s:w:r:t:c:h")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_dir = optarg;
                break;
            case 'v':
                val_file = optarg;
                break;
            case 'g':
                split(img_hw, optarg, ",");
                img_h = ( int )img_hw[0];
                img_w = ( int )img_hw[1];
                break;
            case 's':
                split(scale, optarg, ",");
                break;
            case 'w':
                split(mean, optarg, ",");
                break;
            case 'r':
                loop_count = atoi(optarg);
                break;
            case 't':
                num_thread = atoi(optarg);
                break;
            case 'c':
                swapRB = atoi(optarg);
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

    if (image_dir == nullptr)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (val_file == nullptr)
    {
        fprintf(stderr, "Error: Val file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_dir) || !check_file_exist(val_file))
        return -1;

    if (img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        fprintf(stderr, "Image height not specified, use default %d\n", img_h);
    }

    if (img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        fprintf(stderr, "Image width not specified, use default  %d\n", img_w);
    }

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        scale[0] = DEFAULT_SCALE1;
        scale[1] = DEFAULT_SCALE2;
        scale[2] = DEFAULT_SCALE3;
        fprintf(stderr, "Scale value not specified, use default  %.1f, %.1f, %.1f\n", scale[0], scale[1], scale[2]);
    }

    if (mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        fprintf(stderr, "Mean value not specified, use default   %.1f, %.1f, %.1f\n", mean[0], mean[1], mean[2]);
    }

    if (tengine_classify_accurary(model_file, image_dir, val_file, img_h, img_w, mean, scale, swapRB, loop_count, num_thread) < 0)
        return -1;

    return 0;
}
