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
 * 
 * original model: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
 */

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

const char* score_pred_name[] = {
    "score_8", "score_16", "score_32"};
const char* bbox_pred_name[] = {
    "bbox_8", "bbox_16", "bbox_32"};
const char* kps_pred_name[] = {
    "kps_8", "kps_16", "kps_32"};
bool has_kps = true;
struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
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

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
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
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

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

static void generate_proposals_f(int feat_stride, const float* score_blob,
                                 const float* bbox_blob, const float* kps_blob,
                                 float prob_threshold, std::vector<FaceObject>& faceobjects, int letterbox_cols, int letterbox_rows)
{
    static float anchors[] = {-8.f, -8.f, 8.f, 8.f, -16.f, -16.f, 16.f, 16.f, -32.f, -32.f, 32.f, 32.f, -64.f, -64.f, 64.f, 64.f, -128.f, -128.f, 128.f, 128.f, -256.f, -256.f, 256.f, 256.f};
    int feat_w = letterbox_cols / feat_stride;
    int feat_h = letterbox_rows / feat_stride;
    int feat_size = feat_w * feat_h;
    int anchor_group = 1;
    if (feat_stride == 8)
        anchor_group = 1;
    if (feat_stride == 16)
        anchor_group = 2;
    if (feat_stride == 32)
        anchor_group = 3;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = 2;

    for (int q = 0; q < num_anchors; q++)
    {
        // shifted anchor
        float anchor_y = anchors[(anchor_group - 1) * 8 + q * 4 + 1];

        float anchor_w = anchors[(anchor_group - 1) * 8 + q * 4 + 2] - anchors[(anchor_group - 1) * 8 + q * 4 + 0];
        float anchor_h = anchors[(anchor_group - 1) * 8 + q * 4 + 3] - anchors[(anchor_group - 1) * 8 + q * 4 + 1];

        for (int i = 0; i < feat_h; i++)
        {
            float anchor_x = anchors[(anchor_group - 1) * 8 + q * 4 + 0];

            for (int j = 0; j < feat_w; j++)
            {
                int index = i * feat_w + j;

                float prob = score_blob[q * feat_size + index];

                if (prob >= prob_threshold)
                {
                    // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                    float dx = bbox_blob[(q * 4 + 0) * feat_size + index] * feat_stride;
                    float dy = bbox_blob[(q * 4 + 1) * feat_size + index] * feat_stride;
                    float dw = bbox_blob[(q * 4 + 2) * feat_size + index] * feat_stride;
                    float dh = bbox_blob[(q * 4 + 3) * feat_size + index] * feat_stride;
                    // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    if (kps_blob != 0)
                    {
                        obj.landmark[0].x = cx + kps_blob[index] * feat_stride;
                        obj.landmark[0].y = cy + kps_blob[1 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[1].x = cx + kps_blob[2 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[1].y = cy + kps_blob[3 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[2].x = cx + kps_blob[4 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[2].y = cy + kps_blob[5 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[3].x = cx + kps_blob[6 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[3].y = cy + kps_blob[7 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[4].x = cx + kps_blob[8 * feat_h * feat_w + index] * feat_stride;
                        obj.landmark[4].y = cy + kps_blob[9 * feat_h * feat_w + index] * feat_stride;
                    }

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<FaceObject>& objects)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const FaceObject& obj = objects[i];

        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        if (has_kps)
        {
            cv::circle(image, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

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

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imwrite("scrfd_out.jpg", image);
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

void get_input_data_scrfd(const char* image_file, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
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

    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_8UC3, cv::Scalar(0, 0, 0));
    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;

    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    img_new.convertTo(img_new, CV_32FC3);
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

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;

    int img_c = 3;
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float scale[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};

    // allow none square letterbox, set default letterbox size
    int letterbox_rows = 640;
    int letterbox_cols = 640;

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
    get_input_data_scrfd(image_file, input_data.data(), letterbox_rows, letterbox_cols, mean, scale);

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
        min_time = (std::min)(min_time, cur);
        max_time = (std::max)(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* postprocess */
    const float prob_threshold = 0.3f;
    const float nms_threshold = 0.45f;

    std::vector<FaceObject> proposals, objects;
    int strides[] = {8, 16, 32};
    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        tensor_t score_tensor = get_graph_tensor(graph, score_pred_name[stride_index]);
        tensor_t bbox_tensor = get_graph_tensor(graph, bbox_pred_name[stride_index]);
        tensor_t kps_tensor = get_graph_tensor(graph, kps_pred_name[stride_index]);
        if (NULL == score_tensor || NULL == bbox_tensor || NULL == kps_tensor)
        {
            fprintf(stderr, "get graph tensor failed\n");
            return -1;
        }
        const float* score_pred = (const float*)get_tensor_buffer(score_tensor);
        const float* bbox_pred = (const float*)get_tensor_buffer(bbox_tensor);
        const float* kps_pred = (const float*)get_tensor_buffer(kps_tensor);

        std::vector<FaceObject> objects_temp;
        generate_proposals_f(strides[stride_index], score_pred, bbox_pred, kps_pred, prob_threshold, objects_temp, letterbox_rows, letterbox_cols);

        proposals.insert(proposals.end(), objects_temp.begin(), objects_temp.end());
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    /* draw the result */

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

    int count = picked.size();
    fprintf(stderr, "detection num: %d\n", count);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        x0 = (std::max)((std::min)(x0, (float)(img.cols - 1)), 0.f);
        y0 = (std::max)((std::min)(y0, (float)(img.rows - 1)), 0.f);
        x1 = (std::max)((std::min)(x1, (float)(img.cols - 1)), 0.f);
        y1 = (std::max)((std::min)(y1, (float)(img.rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        if (has_kps)
        {
            float x0 = (objects[i].landmark[0].x - tmp_w) * ratio_x;
            float y0 = (objects[i].landmark[0].y - tmp_h) * ratio_y;
            float x1 = (objects[i].landmark[1].x - tmp_w) * ratio_x;
            float y1 = (objects[i].landmark[1].y - tmp_h) * ratio_y;
            float x2 = (objects[i].landmark[2].x - tmp_w) * ratio_x;
            float y2 = (objects[i].landmark[2].y - tmp_h) * ratio_y;
            float x3 = (objects[i].landmark[3].x - tmp_w) * ratio_x;
            float y3 = (objects[i].landmark[3].y - tmp_h) * ratio_y;
            float x4 = (objects[i].landmark[4].x - tmp_w) * ratio_x;
            float y4 = (objects[i].landmark[4].y - tmp_h) * ratio_y;

            objects[i].landmark[0].x = (std::max)((std::min)(x0, (float)(img.cols - 1)), 0.f);
            objects[i].landmark[0].y = (std::max)((std::min)(y0, (float)(img.rows - 1)), 0.f);
            objects[i].landmark[1].x = (std::max)((std::min)(x1, (float)(img.cols - 1)), 0.f);
            objects[i].landmark[1].y = (std::max)((std::min)(y1, (float)(img.rows - 1)), 0.f);
            objects[i].landmark[2].x = (std::max)((std::min)(x2, (float)(img.cols - 1)), 0.f);
            objects[i].landmark[2].y = (std::max)((std::min)(y2, (float)(img.rows - 1)), 0.f);
            objects[i].landmark[3].x = (std::max)((std::min)(x3, (float)(img.cols - 1)), 0.f);
            objects[i].landmark[3].y = (std::max)((std::min)(y3, (float)(img.rows - 1)), 0.f);
            objects[i].landmark[4].x = (std::max)((std::min)(x4, (float)(img.cols - 1)), 0.f);
            objects[i].landmark[4].y = (std::max)((std::min)(y4, (float)(img.rows - 1)), 0.f);
        }
    }

    draw_objects(img, objects);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
