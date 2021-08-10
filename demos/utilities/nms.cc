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


int nms(const std::vector<Region>& before, std::vector<Region>& after, const float& nms_threshold)
{
    // clear out
    after.clear();
    std::vector<int> picked;

    const auto n = before.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = before[i].box.width * before[i].box.height;
    }

    for (int i = 0; i < n; i++)
    {
        const auto& a = before[i];

        int keep = 1;
        for (auto& val : picked)
        {
            const auto& b = before[val];

            float iou_val = iou(cv::Rect2f(a.box.x, a.box.y, a.box.width, a.box.height), cv::Rect2f(b.box.x, b.box.y, b.box.width, b.box.height));

            if (iou_val > nms_threshold)
                keep = 0;
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    for (auto& val : picked)
    {
        after.push_back(before[val]);
    }

    return 0;
};


int nms(const std::vector<Face>& before, std::vector<Face>& after, const float& nms_threshold)
{
    // clear out
    after.clear();
    std::vector<int> picked;

    const auto n = before.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = before[i].box.width * before[i].box.height;
    }

    for (int i = 0; i < n; i++)
    {
        const auto& a = before[i];

        int keep = 1;
        for (auto& val : picked)
        {
            const auto& b = before[val];

            float iou_val = iou(cv::Rect2f(a.box.x, a.box.y, a.box.width, a.box.height), cv::Rect2f(b.box.x, b.box.y, b.box.width, b.box.height));

            if (iou_val > nms_threshold)
                keep = 0;
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }

    for (auto& val : picked)
    {
        after.push_back(before[val]);
    }

    return 0;
};
