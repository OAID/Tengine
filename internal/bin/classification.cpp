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
 * Copyright (c) 2018, Open AI Lab
 * Author: youj@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <algorithm>

#include "tengine_operations.h"
#include "tengine_c_api.h"

#define DEFAULT_REPEAT_CNT 1
#define PRINT_TOP_NUM 5

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float>& v, const int N)
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

void get_input_data(const char* image_file, float* input_data, const int img_h, const int img_w, const float* mean,
                    const float scale)
{
    image im = imread(image_file);
    image resImg = resize_image(im, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);

    float* img_data = ( float* )resImg.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
    {
        for(int h = 0; h < img_h; h++)
        {
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
        }
    }
}

void get_input_data_tf(const char* image_file, float* input_data, const int img_h, const int img_w, const float mean,
                       const float scale)
{
    image im = imread(image_file);
    image resImg = resize_image(im, img_w, img_h);
    for(int i = 0; i < 3 * img_w * img_h; i++)
        resImg.data[i] = (resImg.data[i] - mean) * scale;

    memcpy(input_data, resImg.data, img_h * img_w * sizeof(float) * 3);
}

void get_input_data_uint8(const char* image_file, uint8_t* input_data, int img_h, int img_w)
{
    image im = imread(image_file);
    image resImg = resize_image(im, img_w, img_h);

    float* img_data = ( float* )resImg.data;
    uint8_t* ptr = input_data;

    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                *ptr = img_data[2 - c];
                ptr++;
            }
            img_data += 3;
        }
    }
}

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void PrintTopLabels(const char* label_file, float* data)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float* end = data + 1000;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, PRINT_TOP_NUM);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
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

bool run_tengine_library(const char* model_format, const int model_fnum, const char* proto_file, const char* model_file,
                         const char* label_file, const char* image_file, const int img_h, const int img_w,
                         const float* mean, const float scale, const int repeat_count)
{
    bool tf_format = false;
    bool tflite_format = false;
    if(strcmp(model_format, "tensorflow") == 0)
        tf_format = true;
    else if(strcmp(model_format, "tflite") == 0)
        tflite_format = true;

    std::string model_name = "model1";

    // init tengine
    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return false;

    // load model
    graph_t graph = nullptr;
    if(model_fnum == 2)
    {
        graph = create_graph(nullptr, model_format, proto_file, model_file);
    }
    else    // model_fnum == 1
    {
        graph = create_graph(nullptr, model_format, model_file);
    }

    if(!graph)
    {
        std::cerr << "Create graph0 failed.\n";
        std::cout << "errno:" << get_tengine_errno() << "\n";
        return false;
    }

    // input
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};
    int dims_tflite[] = {1, img_h, img_w, 3};

    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    uint8_t* input_data_uint8 = ( uint8_t* )malloc(sizeof(uint8_t) * img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(tflite_format || tf_format)
        set_tensor_shape(input_tensor, dims_tflite, 4);
    else
        set_tensor_shape(input_tensor, dims, 4);

    // prerun
    int ret = prerun_graph(graph);
    if(-1 == ret)
    {
        std::cout << " prerun graph failed \n";
        return false;
    }
    // dump_graph(graph);

    struct timeval t0, t1;
    float avg_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        if(tf_format)
        {
            get_input_data_tf(image_file, input_data, img_h, img_w, mean[0], scale);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);
        }
        else if(tflite_format)
        {
            get_input_data_uint8(image_file, input_data_uint8, img_h, img_w);
            set_tensor_buffer(input_tensor, input_data_uint8, img_size * 4);
        }
        else
        {
            get_input_data(image_file, input_data, img_h, img_w, mean, scale);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);
        }

        gettimeofday(&t0, NULL);
        run_graph(graph, 1);
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    if(model_fnum == 2)
        std::cout << "\nProto file : " << proto_file;
    std::cout << "\nModel file : " << model_file << "\n"
              << "label file : " << label_file << "\n"
              << "image file : " << image_file << "\n"
              << "img_h, img_w, scale, mean[3] : " << img_h << ", " << img_w << ", " << scale << ", " << mean[0] << ", "
              << mean[1] << ", " << mean[2] << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    // print output
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* data = ( float* )get_tensor_buffer(output_tensor);
    uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(output_tensor);
    int data_size = get_tensor_buffer_size(output_tensor);

    float output_scale = 0.0f;
    int zero_point = 0;
    get_tensor_quant_param(output_tensor, &output_scale, &zero_point, 1);
    // std::cout << "output scale: " << output_scale << " , zero: " << zero_point << "\n";

    if(tflite_format)
        PrintTopLabels_uint8(label_file, data_uint8, data_size, output_scale, zero_point);
    else
        PrintTopLabels(label_file, data);
    std::cout << "--------------------------------------\n";

    free(input_data);
    free(input_data_uint8);
    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();

    return true;
}

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

static bool check_file_exist(const std::string file_name)
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

static bool check_params(const std::string model_format, int& model_fnum, const std::string proto_file,
                         const std::string model_file, const std::string label_file, const std::string image_file,
                         const int img_h, const int img_w, const float scale, const int repeat_count)
{
    // check model format
    if(model_format.empty())
    {
        std::cerr << "Model format not specified.\n";
        return false;
    }
    else if(model_format == "caffe" || model_format == "mxnet")
    {
        model_fnum = 2;
        if(proto_file.empty() || model_file.empty())
        {
            std::cerr << "Both proto file and model file should be specified.\n";
            return false;
        }
    }
    else if(model_format == "caffe_single" || model_format == "onnx" || model_format == "tensorflow" ||
            model_format == "tflite")
    {
        model_fnum = 1;
        if(model_file.empty())
        {
            std::cerr << "Model file should be specified.\n";
            return false;
        }
    }
    else
    {
        std::cerr << "Model format not supported: " << model_format << "\n";
        return false;
    }

    // check input files
    if((model_fnum == 2 && !check_file_exist(proto_file)) || !check_file_exist(model_file) ||
       !check_file_exist(label_file) || !check_file_exist(image_file))
    {
        return false;
    }

    // check other params
    if(img_h <= 0 || img_w <= 0 || scale <= 0 || repeat_count <= 0)
    {
        std::cerr << "Invalid input params.\n";
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    // params
    int repeat_count = DEFAULT_REPEAT_CNT;

    std::string model_format;
    std::string proto_file;
    std::string model_file;
    std::string label_file;
    std::string image_file;
    int model_fnum = 0;

    int img_h = 224;
    int img_w = 224;
    float scale = 1.f;
    float mean[3] = {104.007, 116.669, 122.679};

    std::vector<int> hw;
    std::vector<float> ms;
    int res;

    if(argc == 1)
    {
        std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                  << "    [-f model_format] [-p proto_file] [-m model_file]\n"
                  << "    [-l label_file] [-i image_file] [-g img_h,img_w]\n"
                  << "    [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";
        return 0;
    }
    while((res = getopt(argc, argv, "f:p:m:l:i:g:s:w:r:h")) != -1)
    {
        switch(res)
        {
            case 'f':
                model_format = optarg;
                break;
            case 'p':
                proto_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'l':
                label_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'g':
                hw = ParseString<int>(optarg);
                if(hw.size() != 2)
                {
                    std::cerr << "Error -g parameter.\n";
                    return -1;
                }
                img_h = hw[0];
                img_w = hw[1];
                break;
            case 's':
                scale = strtof(optarg, NULL);
                break;
            case 'w':
                ms = ParseString<float>(optarg);
                if(ms.size() != 3)
                {
                    std::cerr << "Error -w parameter.\n";
                    return -1;
                }
                mean[0] = ms[0];
                mean[1] = ms[1];
                mean[2] = ms[2];
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "    [-f model_format] [-p proto_file] [-m model_file]\n"
                          << "    [-l label_file] [-i image_file] [-g img_h,img_w]\n"
                          << "    [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";
                return 0;
            default:
                break;
        }
    }

    // check all the params
    if(!check_params(model_format, model_fnum, proto_file, model_file, label_file, image_file, img_h, img_w, scale,
                     repeat_count))
        return -1;

    // start to run
    if(run_tengine_library(model_format.c_str(), model_fnum, proto_file.c_str(), model_file.c_str(), label_file.c_str(),
                           image_file.c_str(), img_h, img_w, mean, scale, repeat_count))
        std::cout << "ALL TEST DONE\n";

    return 0;
}
