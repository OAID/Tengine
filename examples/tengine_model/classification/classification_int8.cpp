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
 * Author: jingyou@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <math.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "common.hpp"
#include "cpu_device.h"

#define DEFAULT_MODEL_NAME "squeezenet"
#define DEFAULT_IMAGE_FILE "tests/images/cat.jpg"
#define DEFAULT_LABEL_FILE "models/synset_words.txt"
#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_REPEAT_CNT 1
#define PRINT_TOP_NUM 5

typedef struct
{
    const char* model_name;
    int img_h;
    int img_w;
    float scale;
    float mean[3];
    const char* tm_file;
    const char* label_file;
} Model_Config;

const Model_Config model_list[] = {
    {"squeezenet", 227, 227, 1.f, {104.007, 116.669, 122.679}, "squeezenet_int8.tmfile", "synset_words.txt"},
    {"mobilenet", 224, 224, 0.017, {104.007, 116.669, 122.679}, "mobilenet_int8.tmfile", "synset_words.txt"},
    {"mobilenet_v2", 224, 224, 0.017, {104.007, 116.669, 122.679}, "mobilenet_v2_int8.tmfile", "synset_words.txt"},
    {"resnet50", 224, 224, 1.f, {104.007, 116.669, 122.679}, "resnet50_int8.tmfile", "synset_words.txt"},
    {"alexnet", 227, 227, 1.f, {104.007, 116.669, 122.679}, "alexnet_int8.tmfile", "synset_words.txt"},
    {"googlenet", 224, 224, 1.f, {104.007, 116.669, 122.679}, "googlenet_int8.tmfile", "synset_words.txt"},
    {"inception_v3", 395, 395, 0.0078, {104.007, 116.669, 122.679}, "inception_v3_int8.tmfile", "synset2015.txt"},
    {"inception_v4", 299, 299, 1 / 127.5f, {104.007, 116.669, 122.679}, "inception_v4_int8.tmfile", "synset_words.txt"},
    {"vgg16", 224, 224, 1.f, {104.007, 116.669, 122.679}, "vgg16_int8.tmfile", "synset_words.txt"}
};


const Model_Config* get_model_config(const char* model_name)
{
    std::string name1 = model_name;
    for(unsigned int i = 0; i < name1.size(); i++)
        name1[i] = tolower(name1[i]);

    for(unsigned int i = 0; i < sizeof(model_list) / sizeof(Model_Config); i++)
    {
        std::string name2 = model_list[i].model_name;
        if(name1 == name2)
        {
            return &model_list[i];
        }
    }
    std::cerr << "Not support model name : " << model_name << "\n";
    return nullptr;
}

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

static inline bool PairCompare(const std::pair<int8_t, int>& lhs, const std::pair<int8_t, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<int8_t>& v, int N)
{
    std::vector<std::pair<int8_t, int>> pairs;
    for(size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for(int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

static float get_absmax_val(float* data, int data_size)
{
    float max_val = 0.f;
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

void get_input_data(const char* image_file, int8_t* input_data, int img_h, int img_w, const float* mean, float scale,
                    float *input_scale, int *zero_point)
{
    cv::Mat sample = cv::imread(image_file, -1);
    if(sample.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
    cv::Mat img;
    if(sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if(sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;

    float* temp_data = (float*)malloc(hw*3*sizeof(float));
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                temp_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
        }
    }

    float input_max = get_absmax_val(temp_data, hw*3);
    *input_scale = input_max / 127;
    *zero_point = 0;

    for(int i = 0; i < hw*3; i++)
        input_data[i] = (int8_t)(round(temp_data[i] / *input_scale) + *zero_point);

    free(temp_data);
}

void PrintTopLabels(const char* label_file, int8_t* data, float q_scale)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    int8_t* end = data + 1000;
    std::vector<int8_t> result(data, end);
    std::vector<int> top_N = Argmax(result, PRINT_TOP_NUM);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        float val = result[idx] * q_scale;
        std::cout << std::fixed << std::setprecision(4) << val << " - \"" << labels[idx] << "\"\n";
    }
}

bool run_tengine_library(const char* model_name, const char* tm_file, const char* label_file, const char* image_file,
                         int img_h, int img_w, const float* mean, float scale, int repeat_count)
{
    // init
    init_tengine();
    if(request_tengine_version("1.2") < 0)
        return false;

    // create graph
    graph_t graph = create_graph(nullptr, "tengine", tm_file);
    if(graph == nullptr)
    {
        std::cerr << "Create graph failed.\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return false;
    }

    // set input shape
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};
    int8_t* input_data = ( int8_t* )malloc(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::cerr << "Get input tensor failed\n";
        return false;
    }
    set_tensor_shape(input_tensor, dims, 4);

    // prerun
    if(prerun_graph(graph) < 0)
    {
        std::cerr << "Prerun graph failed\n";
        return false;
    }
    //dump_graph(graph);

    struct timeval t0, t1;
    float avg_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        float input_scale;
        int zero_point;
        get_input_data(image_file, input_data, img_h, img_w, mean, scale, &input_scale, &zero_point);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        set_tensor_quant_param(input_tensor, &input_scale, &zero_point, 1);

        gettimeofday(&t0, NULL);
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return false;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "\nModel name : " << model_name << "\n"
              << "tengine model file : " << tm_file << "\n"
              << "label file : " << label_file << "\n"
              << "image file : " << image_file << "\n"
              << "img_h, imag_w, scale, mean[3] : " << img_h << " " << img_w << " " << scale << " " << mean[0] << " "
              << mean[1] << " " << mean[2] << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    // print output
    float q_scale;
    int q_zero;
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    get_tensor_quant_param(output_tensor, &q_scale, &q_zero, 1);
    int8_t* data = ( int8_t* )get_tensor_buffer(output_tensor);
    PrintTopLabels(label_file, data, q_scale);
    std::cout << "--------------------------------------\n";

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

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

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_CNT;
    const std::string root_path = get_root_path();
    std::string model_name;
    std::string tm_file;
    std::string label_file;
    std::string image_file;
    std::vector<int> hw;
    std::vector<float> ms;
    int img_h = 0;
    int img_w = 0;
    float scale = 0.0;
    float mean[3] = {-1.0, -1.0, -1.0};

    int res;
    while((res = getopt(argc, argv, "n:t:l:i:g:s:w:r:h")) != -1)
    {
        switch(res)
        {
            case 'n':
                model_name = optarg;
                break;
            case 't':
                tm_file = optarg;
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
                          << "    [-n model_name] [-t tm_file] [-l label_file] [-i image_file]\n"
                          << "    [-g img_h,img_w] [-s scale] [-w mean[0],mean[1],mean[2]] [-r repeat_count]\n";
                return 0;
            default:
                break;
        }
    }

    const Model_Config* mod_config;
    // if model files not specified
    if(tm_file.empty())
    {
        // if model name not specified
        if(model_name.empty())
        {
            // use default model
            model_name = DEFAULT_MODEL_NAME;
            std::cout << "Model name and tm file not specified, run " << model_name << " by default.\n";
        }
        // get model config in predefined model list
        mod_config = get_model_config(model_name.c_str());
        if(mod_config == nullptr)
            return -1;

        // get tm file
        tm_file = get_file(mod_config->tm_file);
        if(tm_file.empty())
            return -1;

        // if label file not specified
        if(label_file.empty())
        {
            // get label file
            label_file = get_file(mod_config->label_file);
            if(label_file.empty())
                return -1;
        }

        if(!hw.size())
        {
            img_h = mod_config->img_h;
            img_w = mod_config->img_w;
        }
        if(scale == 0.0)
            scale = mod_config->scale;
        if(!ms.size())
        {
            mean[0] = mod_config->mean[0];
            mean[1] = mod_config->mean[1];
            mean[2] = mod_config->mean[2];
        }
    }

    // if label file not specified, use default label file
    if(label_file.empty())
    {
        label_file = root_path + DEFAULT_LABEL_FILE;
        std::cout << "Label file not specified, use " << label_file << " by default.\n";
    }

    // if image file not specified, use default image file
    if(image_file.empty())
    {
        image_file = root_path + DEFAULT_IMAGE_FILE;
        std::cout << "Image file not specified, use " << image_file << " by default.\n";
    }

    if(img_h == 0)
        img_h = DEFAULT_IMG_H;
    if(img_w == 0)
        img_w = DEFAULT_IMG_W;
    if(scale == 0.0)
        scale = DEFAULT_SCALE;
    if(mean[0] == -1.0)
        mean[0] = DEFAULT_MEAN1;
    if(mean[1] == -1.0)
        mean[1] = DEFAULT_MEAN2;
    if(mean[2] == -1.0)
        mean[2] = DEFAULT_MEAN3;
    if(model_name.empty())
        model_name = "unknown";

    // check input files
    if(!check_file_exist(tm_file) || !check_file_exist(label_file) || !check_file_exist(image_file))
        return -1;

    // start to run
    if(!run_tengine_library(model_name.c_str(), tm_file.c_str(), label_file.c_str(), image_file.c_str(), img_h, img_w,
                            mean, scale, repeat_count))
        return -1;

    std::cout << "ALL TEST DONE\n";

    return 0;
}
