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

#include "permute.hpp"

bool permute(const cv::Mat& src, cv::Mat& dst, bool swap_rb)
{
    if (src.empty() || dst.empty() || dst.data == src.data || dst.rows != src.rows || dst.cols != src.cols)
    {
        return false;
    }

    std::vector<cv::Mat> permute_vector(src.channels());
    for (int i = 0; i < src.channels(); i++)
    {
        uint8_t* ptr = (uint8_t*)(dst.data) + src.cols * src.rows * i;
        cv::Mat channel(src.rows, src.cols, CV_8UC1, ptr);

        if (swap_rb)
        {
            permute_vector[2 - i] = channel;
        }
        else
        {
            permute_vector[i] = channel;
        }
    }

    cv::split(src, permute_vector);

    return true;
}
