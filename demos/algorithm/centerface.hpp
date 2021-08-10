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

class CenterFace
{
public:
    CenterFace();
    ~CenterFace();
    bool Load(const std::string& detection_model, const cv::Size& input_shape, const std::string& device);
    bool Detect(const cv::Mat& image, std::vector<Region>& boxes, const float& score_threshold, const float& nms_threshold);

private:
    bool Init(int width, int height, int channel);
    void decode_heatmap(const int& image_width, const int& image_height, const float& threshold, std::vector<Region>& boxes);

private:
    context_t context;

    graph_t graph;

    std::vector<uint8_t> canvas_uint8;
    std::vector<float> canvas_float;

    std::vector<uint8_t> resized_container;
    std::vector<uint8_t> resized_permute_container;

    std::vector<float> score_heatmap;
    std::vector<float> bbox_heatmap;

    std::vector<Region> region_heatmap;

    float heatmap_threshold;
    float nms_threshold;

    float width_gap;
    float height_gap;
    int canvas_width;
    int canvas_height;

    int heatmap_width;
    int heatmap_height;

    float input_scale;
    int input_zp;

    float score_scale;
    float bbox_scale;
    int score_zp;
    int bbox_zp;
};
