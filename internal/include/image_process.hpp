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

#include "tengine_operations.h"
#include <cstdlib>
#include <cstdio>

namespace TEngine {

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    image img = imread(image_file);
    img = rgb2bgr_premute(img);
    image resImg = resize_image(img, img_w, img_h);

    // int index = 0;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (resImg.data[c * hw + h * img_w + w] - mean[c]) * scale;
            }
        }
    }
}    // namespace TEngine

}    // namespace TEngine

#endif
