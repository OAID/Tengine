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

class recognition
{
public:
    recognition() = default;
    ~recognition();

public:
    bool load(const std::string& model, const std::string& device);
    bool get_feature(const cv::Mat& image, const Coordinate landmark[5], std::vector<float>& feature);
    bool get_feature_std(const cv::Mat& image, std::vector<float>& feature);

private:
    context_t context;
    graph_t graph;

private:
    tensor_t input_tensor;
    tensor_t output_tensor;

private:
    float input_scale;
    float output_scale;
    int input_zp;
    int output_zp;

private:
    std::vector<uint8_t> input_affine;
    std::vector<uint8_t> input_uint8_buffer;
    std::vector<float> input_float_buffer;
    std::vector<uint8_t> output_uint8_buffer;
    std::vector<float> output_float_buffer;
};
