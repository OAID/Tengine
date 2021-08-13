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

#pragma once

#include "types.hpp"

#include "yolo_layer.hpp"
#include "tengine/c_api.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class YOLO
{
public:
    YOLO(const std::string& model, const int& w, const int& h, const std::array<float, 3>& scale, const std::array<float, 3>& bias);
    ~YOLO();
    int detect(const cv::Mat& image, std::vector<Object>& objects);

private:
    int init();
    void run_post(int image_width, int image_height, std::vector<Object>& boxes);

private:
    context_t context;
    graph_t graph;

    int width;
    int height;

    std::vector<uint8_t> input_uint8;
    std::vector<float> input_float;

    std::vector<std::vector<float> > output_float;

    std::vector<uint8_t> canvas;
    std::vector<uint8_t> canvas_permute;

    float in_scale;
    int in_zp;

    bool init_done;

    std::vector<float> out_scale;
    std::vector<int> out_zp;

private:
    std::array<float, 3> scale;
    std::array<float, 3> bias;
};
