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
 * Author: jxyang@openailab.com
 */
#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <vector>
#include <sstream>
#include <algorithm>
#include <math.h>
#include <string.h>

typedef struct
{
    const char* model_name;
    int img_h;
    int img_w;
    float scale;
    float mean[3];
    const char* proto_file;
    const char* model_file;
    const char* label_file;
    const char* image_file;
} Model_Config;

std::vector<int> Argmax(const std::vector<float>& v, int N);

const Model_Config* get_model_config(const Model_Config model_list[], const int model_list_len, const char* model_name);
std::string get_root_path(void);
std::string get_file(const char* fname);
bool check_file_exist(const std::string file_name);

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale);
void get_input_data_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                         float& input_scale, int& zero_point);
void get_input_data_tf(const char* image_file, float* input_data, const int img_h, const int img_w, const float* mean,
                       const float scale);
void get_input_data_tf_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                            float& input_scale, int& zero_point);
void get_input_data_uint8(const char* image_file, uint8_t* input_data, int img_h, int img_w);
void get_input_data_mx(const char* image_file, float* input_data, int img_h, int img_w, const float* mean);

void LoadLabelFile(std::vector<std::string>& result, const char* fname);
void LoadLabelFile_nasnet(std::vector<std::string>& result, const char* fname);
void PrintTopLabels(const char* label_file, float* data, int data_size);
void PrintTopLabels_int8(const char* label_file, int8_t* data, int data_size, float q_scale);
void PrintTopLabels_uint8(const char* label_file, uint8_t* data, int data_size, float scale, int zero_point);
void PrintTopLabels_int8_function(const char* label_file, int8_t* data, int data_size, float q_scale, const char* model_name);

void PrintTopLabels_common(const char* label_file, float* data, int data_size, const char* model_name);
void get_input_data_mx_common(const char* image_file, float* input_data, int img_h, int img_w, const float* mean);
void get_input_data_mx_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                         float& input_scale, int& zero_point);
template <typename T> static std::vector<T> ParseString(const std::string str)
{
    typedef std::string::size_type pos;
    const char delim_ch = ',';
    std::string str_tmp = str;
    std::vector<T> result;
    T t;

    pos delim_pos = str_tmp.find(delim_ch);
    while(delim_pos != std::string::npos)
    {
        std::istringstream ist(str_tmp.substr(0, delim_pos));
        ist >> t;
        result.push_back(t);
        str_tmp.replace(0, delim_pos + 1, "");
        delim_pos = str_tmp.find(delim_ch);
    }
    if(str_tmp.size() > 0)
    {
        std::istringstream ist(str_tmp);
        ist >> t;
        result.push_back(t);
    }

    return result;
}

#endif    // __COMMON_HPP__
