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
 * Author: lswang@openailab.com
 */

#pragma once

#include "types.hpp"

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "tengine/c_api.h"

class SCRFD
{
public:
    SCRFD();
    ~SCRFD();
    bool Load(const std::string& model, const cv::Size& input_shape, const std::string& device);
    bool Detect(const cv::Mat& image, std::vector<Face>& boxes, const float& score_threshold, const float& nms_threshold);

private:
    bool init_canvas(int width, int height, int channel);
    bool init_buffer();
    bool init_anchor();
    bool pre(const cv::Mat& image);
    bool post(std::vector<Face>& box);

private:
    context_t context;
    graph_t graph;
    bool is_quantization;

private:
    float score_threshold;
    float nms_threshold;

private:
    int canvas_width;
    int canvas_height;
    float canvas_width_gap;
    float canvas_height_gap;
    float input_ratio;

private:
    float input_scale;
    int input_zp;

private:
    std::vector<uint8_t> canvas_uint8;
    std::vector<float> canvas_float;

    std::vector<uint8_t> resized_container;
    std::vector<uint8_t> resized_permute_container;

private:
    std::vector<float> score_scale;
    std::vector<float> bbox_scale;
    std::vector<float> landmark_scale;

    std::vector<int> score_zp;
    std::vector<int> bbox_zp;
    std::vector<int> landmark_zp;

    std::vector<std::vector<float> > score_buffer;
    std::vector<std::vector<float> > bbox_buffer;
    std::vector<std::vector<float> > landmark_buffer;

private:
    std::vector<float> ratios;
    std::vector<float> scales;

    std::vector<int> strides;
    std::vector<float> bases;

    std::vector<std::vector<std::array<float, 4> > > anchors;

    std::vector<std::string> score_name;
    std::vector<std::string> bbox_name;
    std::vector<std::string> landmark_name;
};
