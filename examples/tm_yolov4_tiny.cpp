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
 * Author: 942002795@qq.com
 * Update: xwwang@openailab.com
 */

#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void get_input_data_yolov4(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, const float* scale)
{
    cv::Mat sample = cv::imread(image_file, 1);
    cv::Mat img;

    if (sample.channels() == 1)
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);

    /* resize process */
    cv::resize(img, img, cv::Size(img_w, img_h));
    img.convertTo(img, CV_32FC3);
    float* img_data = (float* )img.data;

    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++)
    {   for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index  = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                input_data[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
}

static void generate_proposals(int stride,  const float* feat, float prob_threshold, std::vector<Object>& objects)
{
    static float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    int anchor_num = 3;
    int feat_w = 416 / stride;
    int feat_h = 416 / stride;
    int cls_num = 80;
    int anchor_group = 0;
    if(stride == 16)
        anchor_group = 1;
    if(stride == 32)
        anchor_group = 2;

    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int anchor = 0; anchor <= anchor_num - 1; anchor++)
            {
                int class_index = 0;
                float class_score = -FLT_MAX;
                int channel_size = feat_h * feat_w;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    int score_index = anchor * 85 * channel_size + feat_w * h + w + (s + 5) * channel_size;
                    float score = feat[score_index];
                    if(score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                float box_score = feat[anchor * 85 * channel_size + feat_w * h + w + 4 * channel_size];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if(final_score >= prob_threshold)
                {
                    int dx_index = anchor * 85 * channel_size + feat_w * h + w + 0 * channel_size;
                    int dy_index = anchor * 85 * channel_size + feat_w * h + w + 1 * channel_size;
                    int dw_index = anchor * 85 * channel_size + feat_w * h + w + 2 * channel_size;
                    int dh_index = anchor * 85 * channel_size + feat_w * h + w + 3 * channel_size;

                    float dx = sigmoid(feat[dx_index]);
                    float dy = sigmoid(feat[dy_index]);

                    float dw = feat[dw_index];
                    float dh = feat[dh_index];

                    float anchor_w = anchors[(anchor_group - 1) * 6 + anchor * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + anchor * 2 + 1];

                    float pred_x = (w + dx) * stride;
                    float pred_y = (h + dy) * stride;
                    float pred_w = exp(dw) * anchor_w ;
                    float pred_h = exp(dh) * anchor_h ;

                    float x0 = (pred_x - pred_w * 0.5f);
                    float y0 = (pred_y - pred_h * 0.5f);
                    float x1 = (pred_x + pred_w * 0.5f);
                    float y1 = (pred_y + pred_h * 0.5f);

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj); 
                }
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("yolov4_tiny_out.jpg", image);
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] \n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    int img_h = 416;
    int img_w = 416;
    int img_c = 3;
    const float mean[3] = {0, 0, 0};
    const float scale[3] = {0.003921, 0.003921, 0.003921};

    int repeat_count = 1;
    int num_thread = 1;

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

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
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
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    int img_size = img_h * img_w * img_c;
    int dims[] = {1, 3, img_h, img_w};
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
    get_input_data_yolov4(image_file, input_data, img_h, img_w, mean, scale);

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
            total_time/repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");


    tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
    tensor_t p32_output = get_graph_output_tensor(graph, 0, 0);

    float* p16_data = ( float*)get_tensor_buffer(p16_output);
    float* p32_data = ( float*)get_tensor_buffer(p32_output);

	/* postprocess */
    const float prob_threshold = 0.45f;
    const float nms_threshold = 0.25f;

    std::vector<Object> proposals;
    std::vector<Object> objects16;
    std::vector<Object> objects32;
    std::vector<Object> objects;

    generate_proposals(32, p32_data, prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    generate_proposals(16, p16_data, prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    /* yolov4 tiny draw the result */
    int raw_h = img.rows;
    int raw_w = img.cols;

    float ratio_x = (float)raw_w / img_w;
    float ratio_y = (float)raw_h / img_h;

    int count = picked.size();
    fprintf(stderr, "detection num: %d\n",count);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = x0 * ratio_x;
        y0 = y0 * ratio_y;
        x1 = x1 * ratio_x;
        y1 = y1 * ratio_y;

        x0 = std::max(std::min(x0, (float)(raw_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(raw_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(raw_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(raw_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    draw_objects(img, objects);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
