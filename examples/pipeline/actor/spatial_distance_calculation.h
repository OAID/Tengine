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
 * Copyright (c) 2021
 * Author: tpoisonooo
 */
#pragma once
#include "pipeline/graph/node.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <deque>

namespace pipeline {
// tan(theta/2) = width /2 / distance
class SpatialDistanceCalc : public Node<Param<std::tuple<cv::Mat, cv::Rect> >, Param<cv::Mat> >
{
public:
    SpatialDistanceCalc() = default;

    void exec() override
    {
        std::tuple<cv::Mat, cv::Rect> in;
        if (input<0>()->pop(in))
        {
            auto mat = std::get<0>(in);
            int width = std::get<1>(in).width;
            if (width <= WIDTH_THRESHOLD)
            {
                auto success = output<0>()->try_push(std::move(mat));
                return;
            }
            m_history_width.push_back(width);

            // smooth width
            double avg = std::accumulate(m_history_width.begin(), m_history_width.end(), 0) * 1.0 / m_history_width.size();
            if (m_history_width.size() > MAX_LEN)
            {
                m_history_width.pop_front();
            }

            auto distance = 1. / avg * SCALE;
            char distance_text[64] = {0};
            sprintf(distance_text, "%.2f", distance);
            cv::putText(mat, std::string(distance_text), cv::Point(40, 40), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 0));

            auto success = output<0>()->try_push(std::move(mat));
            if (not success)
            {
                fprintf(stdout, "drop " __FILE__ "\n");
            }
        }
    }

private:
    std::deque<size_t> m_history_width;
    const size_t MAX_LEN = 8;
    const int WIDTH_THRESHOLD = 10;
    const double SCALE = 100. / 0.27;
};

} // namespace pipeline
