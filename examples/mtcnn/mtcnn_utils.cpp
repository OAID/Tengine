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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include "mtcnn_utils.hpp"

void cal_scale_list(int height, int width, int minsize, std::vector<scale_window>& list)
{
    float factor = 0.709;
    int MIN_DET_SIZE = 12;
    int minl = height < width ? height : width;
    float m = ( float )MIN_DET_SIZE / minsize;
    minl = minl * m;
    int factor_count = 0;
    while(minl > MIN_DET_SIZE)
    {
        if(factor_count > 0)
            m = m * factor;
        minl *= factor;
        factor_count++;

        scale_window win;
        win.h = ( int )ceil(height * m);
        ;
        win.w = ( int )ceil(width * m);
        ;
        win.scale = m;
        list.push_back(win);
    }
}

void set_cvMat_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width)
{
    for(int i = 0; i < 3; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
}

void generate_bounding_box(const float* confidence_data, const float* reg_data, float scale, float threshold,
                           int feature_h, int feature_w, std::vector<face_box>& output, bool transposed)
{
    int stride = 2;
    int cellSize = 12;

    int img_h = feature_h;
    int img_w = feature_w;
    int count = img_h * img_w;
    confidence_data += count;

    for(int i = 0; i < count; i++)
    {
        if(*(confidence_data + i) >= threshold)
        {
            int y = i / img_w;
            int x = i - img_w * y;

            float top_x = ( int )((x * stride + 1) / scale);
            float top_y = ( int )((y * stride + 1) / scale);
            float bottom_x = ( int )((x * stride + cellSize) / scale);
            float bottom_y = ( int )((y * stride + cellSize) / scale);

            face_box box;
            box.x0 = top_x;
            box.y0 = top_y;
            box.x1 = bottom_x;
            box.y1 = bottom_y;

            box.score = *(confidence_data + i);

            int c_offset = y * img_w + x;
            int c_size = img_w * img_h;

            if(transposed)
            {
                box.regress[1] = reg_data[c_offset];
                box.regress[0] = reg_data[c_offset + c_size];
                box.regress[3] = reg_data[c_offset + 2 * c_size];
                box.regress[2] = reg_data[c_offset + 3 * c_size];
            }
            else
            {
                box.regress[0] = reg_data[c_offset];
                box.regress[1] = reg_data[c_offset + c_size];
                box.regress[2] = reg_data[c_offset + 2 * c_size];
                box.regress[3] = reg_data[c_offset + 3 * c_size];
            }

            output.push_back(box);
        }
    }
}

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>& output)
{
    output.clear();
    std::sort(input.begin(), input.end(), [](const face_box& a, const face_box& b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for(int i = 0; i < box_num; i++)
    {
        if(merged[i])
            continue;

        output.push_back(input[i]);

        float h0 = input[i].y1 - input[i].y0 + 1;
        float w0 = input[i].x1 - input[i].x0 + 1;

        float area0 = h0 * w0;

        for(int j = i + 1; j < box_num; j++)
        {
            if(merged[j])
                continue;

            float inner_x0 = std::max(input[i].x0, input[j].x0);
            float inner_y0 = std::max(input[i].y0, input[j].y0);

            float inner_x1 = std::min(input[i].x1, input[j].x1);
            float inner_y1 = std::min(input[i].y1, input[j].y1);

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if(inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y1 - input[j].y0 + 1;
            float w1 = input[j].x1 - input[j].x0 + 1;

            float area1 = h1 * w1;

            float score;

            if(type == NMS_UNION)
            {
                score = inner_area / (area0 + area1 - inner_area);
            }
            else
            {
                score = inner_area / std::min(area0, area1);
            }

            if(score > threshold)
                merged[j] = 1;
        }
    }
}
void regress_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        face_box& box = rects[i];

        float h = box.y1 - box.y0 + 1;
        float w = box.x1 - box.x0 + 1;

        box.x0 = box.x0 + w * box.regress[0];
        box.y0 = box.y0 + h * box.regress[1];
        box.x1 = box.x1 + w * box.regress[2];
        box.y1 = box.y1 + h * box.regress[3];
    }
}
void square_boxes(std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        float h = rects[i].y1 - rects[i].y0 + 1;
        float w = rects[i].x1 - rects[i].x0 + 1;

        float l = std::max(h, w);

        rects[i].x0 = rects[i].x0 + (w - l) * 0.5;
        rects[i].y0 = rects[i].y0 + (h - l) * 0.5;
        rects[i].x1 = rects[i].x0 + l - 1;
        rects[i].y1 = rects[i].y0 + l - 1;
    }
}

void padding(int img_h, int img_w, std::vector<face_box>& rects)
{
    for(unsigned int i = 0; i < rects.size(); i++)
    {
        rects[i].px0 = std::max(rects[i].x0, 1.0f);
        rects[i].py0 = std::max(rects[i].y0, 1.0f);
        rects[i].px1 = std::min(rects[i].x1, ( float )img_w);
        rects[i].py1 = std::min(rects[i].y1, ( float )img_h);
    }
}
void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects, float nms_r_thresh)
{
    nms_boxes(input, nms_r_thresh, NMS_UNION, rects);

    regress_boxes(rects);

    square_boxes(rects);

    padding(img_h, img_w, rects);
}
void copy_one_patch(const cv::Mat& img, face_box& input_box, float* data_to, int width, int height)
{
    std::vector<cv::Mat> channels;

    set_cvMat_input_buffer(channels, data_to, height, width);

    cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1), cv::Range(input_box.px0, input_box.px1));

    int pad_top = std::abs(input_box.py0 - input_box.y0);
    int pad_left = std::abs(input_box.px0 - input_box.x0);
    int pad_bottom = std::abs(input_box.py1 - input_box.y1);
    int pad_right = std::abs(input_box.px1 - input_box.x1);

    cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(0));

    cv::resize(chop_img, chop_img, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::split(chop_img, channels);
}
