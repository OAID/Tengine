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

#include "affine.hpp"


bool affine(const cv::Mat& src, cv::Mat& dst, const Coordinate landmark[5])
{
    const cv::Point2f gt_point[5] = { {38.2946, 51.6963}, {73.5318, 51.5014}, {56.0252, 71.7366}, {41.5493, 92.3655}, {70.7299, 92.2041} };

    if (src.empty() || dst.empty() || dst.data == src.data || 112 != dst.rows || 112 != dst.cols)
    {
        return false;
    }

    std::vector<cv::Point2f> src_points_vector(5), dst_points_vector(5);
    for (int i = 0; i < 5; i++)
    {
        src_points_vector[i] = { landmark[i].x, landmark[i].y };
        dst_points_vector[i] = { gt_point[i].x, gt_point[i].y };
    }

    cv::Mat trans_matrix = cv::estimateAffinePartial2D(src_points_vector, dst_points_vector);
    cv::warpAffine(src, dst, trans_matrix, dst.size());

    return true;
}
