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
 * Author: jingyou@openailab.com
 */
#ifndef __CAFFE_MTCNN_UTILS_HPP__
#define __CAFFE_MTCNN_UTILS_HPP__

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

    /* regression scale */
    float regress[4];

    /* padding stuff */
    float px0;
    float py0;
    float px1;
    float py1;

    face_landmark landmark;
};

/* get current time: in us */
unsigned long get_cur_time(void);

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>& output);

void regress_boxes(std::vector<face_box>& rects);

void square_boxes(std::vector<face_box>& rects);

void padding(int img_h, int img_w, std::vector<face_box>& rects);

void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects);

void generate_bounding_box(const float* confidence_data, int confidence_size, const float* reg_data, float scale,
                           float threshold, int feature_h, int feature_w, std::vector<face_box>& output,
                           bool transposed);

void set_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width);

void cal_pyramid_list(int height, int width, int min_size, float factor, std::vector<scale_window>& list);

#endif    // __CAFFE_MTCNN_UTILS_HPP__