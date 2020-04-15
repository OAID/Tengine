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
 * Author: qtang@openailab.com
 */

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>

#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_cpp_api.h"
#include "common_util.hpp"

#include "cmdline.h"
#include "json.hpp"


struct ClassScore { int idx; float score; };


bool read_file(const std::string &file_path, std::vector<uint8_t>& data);

void get_input_data(const char* const image_file, float* const input_data, int img_h, int img_w, const float* const mean, const float* const scale);

// sort class score
bool sort_class_score(const ClassScore& a, const ClassScore&b);

void get_result(float* data, std::vector<ClassScore>& result);

void printf_class_score(std::vector<ClassScore>& result);

bool parse_model(const std::vector<uint8_t>& buffer, std::string& path, std::string& input_name, std::string& output_name, float* const scale, float* shift, int& width, int& height);

bool parse_image_list(const std::vector<uint8_t>& buffer, std::vector<std::string>& image_list);

bool parse_top_n(const std::vector<uint8_t>& buffer, std::vector<std::vector<ClassScore>>& image_topN);
