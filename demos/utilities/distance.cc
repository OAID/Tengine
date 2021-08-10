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
 * Author: lswang@openailab.com
 */

#include "nms.hpp"
#include "iou.hpp"

#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>


float distance(const std::vector<float>& a, std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty() || b.empty())
    {
        return -1.f;
    }

    float sum = 0.f;
    for (int i = 0; i < a.size(); i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}


float cos_distance(const std::vector<float>& a, std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty() || b.empty())
    {
        return -1.f;
    }

    auto reg_norm = 0.f, ver_norm = 0.f, product = distance(a, b);
    for (int i = 0; i < a.size(); i++)
    {
        reg_norm += a[i] * a[i];
        ver_norm += b[i] * b[i];
    }

    return product / (std::sqrt(reg_norm) * std::sqrt(ver_norm));
}


void norm_feature(std::vector<float>& feature)
{
    auto sum = 0.f;
    for (auto& val : feature)
    {
        sum += val * val;
    }

    sum = 1.f / std::sqrt(sum);

    for (auto& val : feature)
    {
        val *= sum;
    }
}
