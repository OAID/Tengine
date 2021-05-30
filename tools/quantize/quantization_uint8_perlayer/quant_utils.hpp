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
 * Author: hhchen@openailab.com
 */
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#include <dirent.h>

#include <string>
#include <cmath>

#include "common.hpp"

#include <tr1/unordered_map>

#include "tengine/c_api.h"
extern "C" 
{
    #include "graph/graph.h"
    #include "graph/subgraph.h"
    #include "graph/node.h"
    #include "graph/tensor.h"
}

#include "operator/prototype/convolution_param.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void get_input_data_cv(const char* image_file, float* input_data, int img_h, int img_w, const float* mean,
                       const float* scale, int img_c, int sw_RGB, int center_crop, int letterbox_rows, int letterbox_cols, int focus);

void readFileList(std::string basePath, std::vector<std::string>& imgs);

double cosin_similarity(float** in_a,float** in_b, uint32_t imgs_num, uint32_t output_num);

std::vector<uint32_t> histCount_int(float *data, uint32_t elem_num, float max_val, float min_val);
std::vector<uint32_t> histCount(float *data, uint32_t elem_num, float max_val, float min_val);

float compute_kl_divergence(std::vector<float> &dist_a, std::vector<float> &dist_b);

std::vector<float> normalize_histogram(std::vector<uint32_t> &histogram);

int threshold_distribution(std::vector<uint32_t> &distribution_in, const int target_bin);
