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

#include "yolo_layer.hpp"

#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;

struct Object_CV
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object_CV& a, const Object_CV& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object_CV>& faceobjects, int left, int right)
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

static void qsort_descent_inplace(std::vector<Object_CV>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object_CV>& faceobjects, std::vector<int>& picked, float nms_threshold)
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
        const Object_CV& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object_CV& b = faceobjects[picked[j]];

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

static void generate_proposals(int net_width, int net_height, int stride, const float* feat, float prob_threshold, std::vector<Object_CV>& objects)
{
    static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

    int anchor_num = 3;
    int feat_w = net_width / stride;
    int feat_h = net_height / stride;
    int cls_num = 80;
    int anchor_group = 0;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;
    //printf("anchor_group:%d\n",anchor_group);
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
                    if (score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                float box_score = feat[anchor * 85 * channel_size + feat_w * h + w + 4 * channel_size];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold)
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
                    float pred_w = exp(dw) * anchor_w;
                    float pred_h = exp(dh) * anchor_h;

                    float x0 = (pred_x - pred_w * 0.5f);
                    float y0 = (pred_y - pred_h * 0.5f);
                    float x1 = (pred_x + pred_w * 0.5f);
                    float y1 = (pred_y + pred_h * 0.5f);

                    Object_CV obj;
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

int pose_process(graph_t graph, int image_width, int image_height, int net_width, int net_height, std::vector<Object>& boxes)
{
    /* dequant output data */
    tensor_t p8_output = get_graph_output_tensor(graph, 2, 0);
    tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
    tensor_t p32_output = get_graph_output_tensor(graph, 0, 0);

    float p8_scale = 0.f;
    float p16_scale = 0.f;
    float p32_scale = 0.f;
    int p8_zero_point = 0;
    int p16_zero_point = 0;
    int p32_zero_point = 0;

    get_tensor_quant_param(p8_output, &p8_scale, &p8_zero_point, 1);
    get_tensor_quant_param(p16_output, &p16_scale, &p16_zero_point, 1);
    get_tensor_quant_param(p32_output, &p32_scale, &p32_zero_point, 1);

    int p8_count = get_tensor_buffer_size(p8_output) / sizeof(uint8_t);
    int p16_count = get_tensor_buffer_size(p16_output) / sizeof(uint8_t);
    int p32_count = get_tensor_buffer_size(p32_output) / sizeof(uint8_t);

    uint8_t* p8_data_u8 = (uint8_t*)get_tensor_buffer(p8_output);
    uint8_t* p16_data_u8 = (uint8_t*)get_tensor_buffer(p16_output);
    uint8_t* p32_data_u8 = (uint8_t*)get_tensor_buffer(p32_output);

    std::vector<float> p8_data(p8_count);
    std::vector<float> p16_data(p16_count);
    std::vector<float> p32_data(p32_count);

    for (int c = 0; c < p8_count; c++)
    {
        p8_data[c] = ((float)p8_data_u8[c] - (float)p8_zero_point) * p8_scale;
    }

    for (int c = 0; c < p16_count; c++)
    {
        p16_data[c] = ((float)p16_data_u8[c] - (float)p16_zero_point) * p16_scale;
    }

    for (int c = 0; c < p32_count; c++)
    {
        p32_data[c] = ((float)p32_data_u8[c] - (float)p32_zero_point) * p32_scale;
    }

    /* postprocess */
    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.25f;

    std::vector<Object_CV> proposals;
    std::vector<Object_CV> objects8;
    std::vector<Object_CV> objects16;
    std::vector<Object_CV> objects32;
    std::vector<Object_CV> objects;

    generate_proposals(net_width, net_height, 32, p32_data.data(), prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    generate_proposals(net_width, net_height, 16, p16_data.data(), prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(net_width, net_height, 8, p8_data.data(), prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((net_height * 1.0 / image_height) < (net_width * 1.0 / image_width))
        scale_letterbox = net_height * 1.0 / image_height;
    else
        scale_letterbox = net_width * 1.0 / image_width;

    resize_rows = int(scale_letterbox * image_height);
    resize_cols = int(scale_letterbox * image_width);

    int tmp_h = (net_height - resize_rows) / 2;
    int tmp_w = (net_width - resize_cols) / 2;

    float ratio_x = (float)image_height / resize_rows;
    float ratio_y = (float)image_width / resize_cols;

    int count = picked.size();
    //    fprintf(stderr, "detection num: %d\n",count);

    boxes.resize(count);
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

        x0 = std::max(std::min(x0, (float)(image_width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(image_height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(image_width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(image_height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        boxes[i].label = objects[i].label;
        boxes[i].score = objects[i].prob;
        boxes[i].box.x = x0;
        boxes[i].box.y = y0;
        boxes[i].box.width = x1 - x0;
        boxes[i].box.height = y1 - y0;
    }

    return 0;
}
