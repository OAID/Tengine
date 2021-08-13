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

#include "letterbox.hpp"

//#define USE_AFFINE_MODE


bool letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Scalar& background_color, float& width_gap, float& height_gap)
{
    if (src.data == dst.data || dst.empty())
    {
        return false;
    }

    const auto target_edge_ratio = (float)dst.size().width / (float)dst.size().height;
    const auto source_edge_ratio = (float)src.size().width / (float)src.size().height;

    // gap in left and right
    if (target_edge_ratio > source_edge_ratio)
    {
        const auto gap_length = (float)dst.size().width - (float)src.size().width * (float)dst.size().height / (float)src.size().height;
        const auto half_gap_length = gap_length / 2.f;

        width_gap  = half_gap_length;
        height_gap = 0.f;

        cv::rectangle(dst, cv::Point2i(0, 0), cv::Point2i((int)width_gap, dst.size().height), background_color);
        cv::rectangle(dst, cv::Point2i(dst.size().width - (int)width_gap, 0), cv::Point2i((int)width_gap, dst.size().height), background_color);
    }
    else
    {
        const auto gap_length = (float)dst.size().height - (float)src.size().height * (float)dst.size().width / (float)src.size().width;
        const auto half_gap_length = gap_length / 2.f;

        width_gap  = 0.f;
        height_gap = half_gap_length;

        cv::rectangle(dst, cv::Point2i(0, 0), cv::Point2i(dst.size().width, (int)height_gap), background_color);
        cv::rectangle(dst, cv::Point2i(0, dst.size().height - (int)height_gap), cv::Point2i(dst.size().width, (int)height_gap), background_color);
    }

#ifdef USE_AFFINE_MODE
    cv::Point2f src_points[3], dst_points[3];

    src_points[0] = { 0.f, 0.f };
    src_points[1] = { (float)src.cols / 2.f, 0.f };
    src_points[2] = { (float)src.cols / 2.f, (float)src.rows / 2.f };

    dst_points[0] = { width_gap, height_gap };
    dst_points[1] = { (float)dst.cols / 2.f, height_gap };
    dst_points[2] = { (float)dst.cols / 2.f, (float)dst.rows / 2.f };


    cv::Mat trans_matrix = cv::getAffineTransform(src_points, dst_points);
    cv::warpAffine(src, dst, trans_matrix, dst.size());

#else
    cv::Mat dst_roi = dst(cv::Rect(width_gap, height_gap, dst.cols - width_gap * 2, dst.rows - height_gap * 2));
    cv::resize(src, dst_roi, dst_roi.size());
#endif

    return true;
}
