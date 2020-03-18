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
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>

#include "tengine_operations.h"
#include "common.hpp"

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline bool PairCompare_int8(const std::pair<int8_t, int>& lhs, const std::pair<int8_t, int>& rhs)
{
    return lhs.first > rhs.first;
}

std::vector<int> Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static inline std::vector<int> Argmax_int8(const std::vector<int8_t>& v, int N)
{
    std::vector<std::pair<int8_t, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare_int8);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

const Model_Config* get_model_config(const Model_Config model_list[], const int model_list_len, const char* model_name)
{
    std::string name1 = model_name;
    for(unsigned int i = 0; i < name1.size(); i++)
        name1[i] = tolower(name1[i]);

    for(int i = 0; i < model_list_len; i++)
    {
        std::string name2 = model_list[i].model_name;
        if(name1 == name2)
        {
            return &model_list[i];
        }
    }
    std::cerr << "Not support model name : " << model_name << "\n";
    return NULL;
}

std::string get_root_path(void)
{
    typedef std::string::size_type pos;
    char buf[1024];

    int rslt = readlink("/proc/self/exe", buf, 1023);
    if(rslt < 0 || rslt > 1023)
    {
        return std::string("");
    }
    buf[rslt] = '\0';

    std::string str = buf;
    std::cout << str << std::endl;
    pos p = str.find("tengine_model_test/");
    if(p != std::string::npos)
        return str.substr(0, p + 19);
    else
        return std::string("");
}

std::string get_file(const char* fname)
{
    FILE* fp;
    std::string fn = fname;

    const std::string mod_sch1 = "./" + fn;
    const std::string mod_sch2 = get_root_path() + "models/" + fn;

    fp = fopen(mod_sch1.c_str(), "r");
    if(fp)
    {
        fclose(fp);
        return mod_sch1;
    }
    else
    {
        fp = fopen(mod_sch2.c_str(), "r");
        if(fp)
        {
            fclose(fp);
            return mod_sch2;
        }
        else
        {
            std::cerr << "Can't find " << fn << " in current dir and models dir.\n";
            return std::string("");
        }
    }
}

bool check_file_exist(const std::string file_name)
{
    FILE* fp = fopen(file_name.c_str(), "r");
    if(!fp)
    {
        std::cerr << "Input file not existed: " << file_name << "\n";
        return false;
    }
    fclose(fp);
    return true;
}

static float get_absmax_val(float* data, int data_size)
{
    float max_val = 0.0f;
    if(data != nullptr)
    {
        for(int i = 0; i < data_size; i++)
        {
            float abs_val = fabs(data[i]);
            if(abs_val > max_val)
                max_val = abs_val;
        }
    }
    return max_val;
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    float means[3] = {mean[0], mean[1], mean[2]};
    float scales[3] = {scale, scale, scale};
    image img = imread(image_file, img_w, img_h, means, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*img.c*img_w*img_h); 
    free_image(img);
}

void get_input_data_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                         float& input_scale, int& zero_point)
{
    int img_size = img_h * img_w * 3;
    float* input_f = (float*)malloc(img_size * sizeof(float));
    get_input_data(image_file, input_f, img_h, img_w, mean, scale);

    float input_max = get_absmax_val(input_f, img_size);
    input_scale = input_max / 127;
    zero_point = 0;
    for(int i = 0; i < img_size; i++)
        input_data[i] = (int8_t)(round(input_f[i] / input_scale) + zero_point);

    free(input_f);
}

void get_input_data_mx_common(const char* image_file, float* input_data, int img_h, int img_w, const float* mean)
{
    float scale[3] = {0.229, 0.224, 0.225};

    float means[3] = {mean[0], mean[1], mean[2]};
    image img = imread(image_file, img_w, img_h, means, scale, MXNET);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h); 
    free_image(img);   
}

void get_input_data_mx_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                         float& input_scale, int& zero_point){
    int img_size = img_h * img_w * 3;
    float* input_f = (float*)malloc(img_size * sizeof(float));
    get_input_data_mx_common(image_file, input_f, img_h, img_w, mean);

    float input_max = get_absmax_val(input_f, img_size);
    input_scale = input_max / 127;
    zero_point = 0;
    for(int i = 0; i < img_size; i++)
        input_data[i] = (int8_t)(round(input_f[i] / input_scale) + zero_point);

    free(input_f);   

}


void get_input_data_tf(const char* image_file, float* input_data, const int img_h, const int img_w, const float* mean,
                       const float scale)
{
    float means[3] = {mean[0], mean[1], mean[2]};
    float scales[3] = {scale, scale, scale};
    image img = imread(image_file, img_w, img_h, means, scales, TENSORFLOW);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);    
    free_image(img);
}

void get_input_data_tf_int8(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                            float& input_scale, int& zero_point)
{
    int img_size = img_h * img_w * 3;
    float* input_f = (float*)malloc(img_size * sizeof(float));
    get_input_data_tf(image_file, input_f, img_h, img_w, mean, scale);

    float input_max = get_absmax_val(input_f, img_size);
    input_scale = input_max / 127;
    zero_point = 0;
    for(int i = 0; i < img_size; i++)
        input_data[i] = (int8_t)(round(input_f[i] / input_scale) + zero_point);

    free(input_f);
}
void get_input_data_uint8(const char* image_file, uint8_t* input_data, int img_h, int img_w)
{
    image im = imread(image_file);

    image resImg = resize_image(im, img_w, img_h);

    int index = 0;
    for(int h = 0; h < img_h; h++)
        for(int w = 0; w < img_w; w++)
            for(int c = 0; c < 3; c++)
                input_data[index++] = ( uint8_t )resImg.data[c * img_h * img_w + h * img_w + w];
    free_image(im);
    free_image(resImg);    

}

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void LoadLabelFile_nasnet(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void PrintTopLabels(const char* label_file, float* data, int data_size)
{
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float* end = data + data_size;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}
void PrintTopLabels_common(const char* label_file, float* data, int data_size, const char* model_name)
{
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float* end = data + data_size;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);
    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        if(strcmp(model_name,"nasnet") == 0 || strcmp(model_name,"inception_resnet_v2") == 0){
            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx-1] << "\"\n";
        } else {
            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
        }        
    }
}
void PrintTopLabels_int8(const char* label_file, int8_t* data, int data_size, float q_scale)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    int8_t* end = data + data_size;
    std::vector<int8_t> result(data, end);
    std::vector<int> top_N = Argmax_int8(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        float val = result[idx] * q_scale;
        std::cout << std::fixed << std::setprecision(4) << val << " - \"" << labels[idx] << "\"\n";
    }
}
void PrintTopLabels_int8_function(const char* label_file, int8_t* data, int data_size, float q_scale, const char* model_name)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    int8_t* end = data + data_size;
    std::vector<int8_t> result(data, end);
    std::vector<int> top_N = Argmax_int8(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        float val = result[idx] * q_scale;
        if(strcmp(model_name,"nasnet") == 0 || strcmp(model_name,"inception_resent_v2") == 0){
            std::cout << std::fixed << std::setprecision(4) << val << " - \"" << labels[idx-1] << "\"\n";
        } else {
            std::cout << std::fixed << std::setprecision(4) << val << " - \"" << labels[idx] << "\"\n";
        }
    }
}
void PrintTopLabels_uint8(const char* label_file, uint8_t* data, int data_size, float scale, int zero_point)
{
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    std::vector<float> result;
    for(int i = 0; i < data_size; i++)
        result.push_back((data[i] - zero_point) * scale);

    std::vector<int> top_N = Argmax(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}

void get_input_data_mx(const char* image_file, float* input_data, int img_h, int img_w, const float* mean)
{
    float scale[3] = {0.229, 0.224, 0.225};   
    float means[3] = {mean[0], mean[1], mean[2]};
    image img = imread(image_file, img_w, img_h, means, scale, MXNET);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);
    free_image(img); 
}
