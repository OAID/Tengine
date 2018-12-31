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
#ifndef __MTCNN_UTILS_HPP__
#define __MTCNN_UTILS_HPP__

#include <iostream>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"

#define NMS_UNION 1
#define NMS_MIN 2

struct scale_window
{
    int h;
    int w;
    float scale;
};

struct face_landmark
{
    float x[5];
    float y[5];
};

struct face_box
{
    float x0;
    float y0;
    float x1;
    float y1;

    /* confidence score */
    float score;

    /*regression scale */
    float regress[4];

    /* padding stuff*/
    float px0;
    float py0;
    float px1;
    float py1;

    face_landmark landmark;
};

void cal_scale_list(int height, int width, int minsize, std::vector<scale_window>& list);

void set_cvMat_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width);

void generate_bounding_box(const float* confidence_data, const float* reg_data, float scale, float threshold,
                           int feature_h, int feature_w, std::vector<face_box>& output, bool transposed);

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>& output);

void regress_boxes(std::vector<face_box>& rects);
void square_boxes(std::vector<face_box>& rects);
void padding(int img_h, int img_w, std::vector<face_box>& rects);
void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects,
                   float nms_r_thresh);
void copy_one_patch(const cv::Mat& img, face_box& input_box, float* data_to, int width, int height);

#endif