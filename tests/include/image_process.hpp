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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __IMAGE_PROCESS_HPP__
#define __IMAGE_PROCESS_HPP__

<<<<<<< HEAD
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstdio>
=======
#include <cstdlib>
#include <cstdio>
#include "tengine_operations.h"
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

namespace TEngine {

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
<<<<<<< HEAD
    cv::Mat img = cv::imread(image_file, -1);

    if(img.empty())
=======
    image img = imread(image_file);

    if(img.data == 0)
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }
<<<<<<< HEAD
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
        for(int w = 0; w < img_w; w++)
            for(int c = 0; c < 3; c++)
=======
    image res_img = resize_image(img, img_h, img_w);
    res_img = rgb2bgr_premute(res_img);
    float* img_data = ( float* )res_img.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
}
}    // namespace TEngine

#endif
