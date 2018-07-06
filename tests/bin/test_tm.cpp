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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "cpu_device.h"

#define DEFAULT_MODEL_DIR        "./models/"
#define DEFAULT_IMAGE_FILE       "./tests/images/cat.jpg"
#define DEFAULT_MODEL_NAME       "squeezenet"

#define DEFAULT_REPEAT_CNT       1
#define PRINT_TOP_NUM            5

#define CREATE_TENGINE_MODEL

typedef struct
{
   const char *model_name;
   int img_h;
   int img_w;
   float scale;
   float mean[3];
   const char *proto_file;
   const char *model_file;
   const char *label_file;
} Model_Config;

const Model_Config model_list[] = {
    { "squeezenet"  ,  227,   227,   1.f,      {104.007, 116.669, 122.679},
      "sqz.prototxt", "squeezenet_v1.1.caffemodel", "synset_words.txt"},
    { "mobilenet"   ,  224,   224,   0.017,    {104.007, 116.669, 122.679},
      "mobilenet_deploy.prototxt", "mobilenet.caffemodel", "synset_words.txt"},
    { "resnet50"    ,  224,   224,   1.f,      {104.007, 116.669, 122.679},
      "resnet50.prototxt", "resnet50.caffemodel", "synset_words.txt"},
    { "alexnet"     ,  227,   227,   1.f,      {104.007, 116.669, 122.679},
      "alex_deploy.prototxt", "alexnet.caffemodel", "synset_words.txt"},
    { "googlenet"   ,  224,   224,   1.f,      {104.007, 116.669, 122.679},
      "googlenet.prototxt", "googlenet.caffemodel", "synset_words.txt"},
    { "inception_v3",  395,   395,   0.0078,   {104.007, 116.669, 122.679},
      "deploy_inceptionV3.prototxt", "deploy_inceptionV3.caffemodel", "synset2015.txt"},
    { "inception_v4",  299,   299,   1/127.5f, {104.007, 116.669, 122.679},
      "inception_v4.prototxt", "inception_v4.caffemodel", "synset_words.txt"},
    { "vgg16"       ,  224,   224,   1.f,      {104.007, 116.669, 122.679},
      "vgg16.prototxt", "vgg16.caffemodel", "synset_words.txt"}
};

const Model_Config * get_model_config(const char *model_name)
{
    std::string name1 = model_name;
    for(unsigned int i=0; i < name1.size(); i++)
        name1[i] = tolower(name1[i]);

    for(unsigned int i=0; i < sizeof(model_list)/sizeof(Model_Config); i++)
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

void LoadLabelFile(std::vector<std::string> &result, const char *fname)
{
    std::ifstream labels(fname);

    std::string line;
    while (std::getline(labels, line))
        result.push_back(line);
}

static inline bool PairCompare(const std::pair<float, int> &lhs,
                               const std::pair<float, int> &rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float> &v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

void get_input_data(const char *image_file, float *input_data, int img_h, int img_w, const float* mean, float scale)
{
    cv::Mat sample = cv::imread(image_file, -1);
    if (sample.empty())
    {
        std::cerr << "Failed to read image file " << image_file << ".\n";
        return;
    }
    cv::Mat img;
    if (sample.channels() == 4) 
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img=sample;
    }
    
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c])*scale;
                img_data++;
            }
        }
    }
}

void PrintTopLabels(const char *label_file, float *data)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float *end = data + 1000;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, PRINT_TOP_NUM);

    for (unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4)
                  << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}


int sys_init(void)
{
    init_tengine_library();
    if(request_tengine_version("0.1")<0)
           return 1;
//    const struct cpu_info * p_info=get_predefined_cpu("rk3399");
//    int cpu_list[]={4,5};
//    set_online_cpu((struct cpu_info *)p_info,cpu_list,sizeof(cpu_list)/sizeof(int));
//    create_cpu_device("rk3399",p_info);

    return 0;
}

bool run_tengine_library(const char *model_name,
                         const char *proto_file, const char *model_file,
                         const char *label_file, const char *image_file,
                         int img_h, int img_w, const float* mean,  float scale,
                         int repeat_count)
{
    // init tengine
    // init_tengine_library();
    // if (request_tengine_version("0.1") < 0)
    //     return false;
    sys_init();

    std::string tm_name = std::string(model_name);
    std::string tm_file = "./tengine_models/" + tm_name + ".tmfile";

#ifdef CREATE_TENGINE_MODEL
    // load model
    if (load_model(model_name, "caffe", proto_file, model_file) < 0)
        return false;
    std::cout << "Load model done.\n";

    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
        return false;
    std::cout << "Create graph0 done.\n";

    // Save the tengine model
    if(save_model(graph, "tengine", tm_file.c_str()) == -1)
        return false;
    std::cout << "Create tengine model file done: "<<tm_file<<"\n";

    destroy_runtime_graph(graph);
    remove_model(model_name);
#endif

    // load the tengine model
    std::string tm_model_name = "tm_" + tm_name;
    if (load_model(tm_model_name.c_str(), "tengine", tm_file.c_str()) < 0)
        return false;
    std::cout << "Load tengine model done.\n";

    // create tengine graph
    graph_t tm_graph = create_runtime_graph("tm_graph0", tm_model_name.c_str(), NULL);
    if (!check_graph_valid(tm_graph))
        return false;
    std::cout << "Create tm_graph0 done.\n";

    // input
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};
    float *input_data = (float *)malloc(sizeof(float) * img_size);

    tensor_t input_tensor = get_graph_input_tensor(tm_graph, 0, 0);
    set_tensor_shape(input_tensor, dims, 4);

    // prerun
    prerun_graph(tm_graph);

    struct timeval t0, t1;
    float avg_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        get_input_data(image_file, input_data, img_h, img_w, mean, scale);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);

        gettimeofday(&t0, NULL);
        run_graph(tm_graph, 1);
        gettimeofday(&t1, NULL);

        float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
    }
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    std::cout << "--------------------------------------\n";

    // print output
    tensor_t output_tensor = get_graph_output_tensor(tm_graph, 0, 0);
    float *data = (float *)get_tensor_buffer(output_tensor);
    PrintTopLabels(label_file, data);
    std::cout << "--------------------------------------\n";

    free(input_data);
    postrun_graph(tm_graph);
    destroy_runtime_graph(tm_graph);
    remove_model(tm_model_name.c_str());

    return true;
}


int main(int argc, char *argv[])
{
    int repeat_count = DEFAULT_REPEAT_CNT;
    const std::string model_path = DEFAULT_MODEL_DIR;
    std::string model_name = DEFAULT_MODEL_NAME;
    std::string proto_file;
    std::string model_file;
    std::string label_file;
    std::string image_file = DEFAULT_IMAGE_FILE;
    std::vector<int> hw;
    std::vector<float> ms;
    int img_h = 0;
    int img_w = 0;
    float scale = 0;
    float mean[3] = {-1, -1, -1};

    int res;
    while((res=getopt(argc,argv,"m:p:r:")) != -1)
    {
        switch(res)
        {
            case 'm':
                model_name = optarg;
                break;
            case 'p':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    const Model_Config * mod_config;
    mod_config = get_model_config(model_name.c_str());
    if(!mod_config)
        return -1;

    proto_file = model_path + mod_config->proto_file;
    model_file = model_path + mod_config->model_file;
    label_file = model_path + mod_config->label_file;
    img_h = mod_config->img_h;
    img_w = mod_config->img_w;
    scale = mod_config->scale;
    mean[0] = mod_config->mean[0];
    mean[1] = mod_config->mean[1];
    mean[2] = mod_config->mean[2];

    // start to run
    if(run_tengine_library(model_name.c_str(),
                           proto_file.c_str(), model_file.c_str(),
                           label_file.c_str(), image_file.c_str(),
                           img_h, img_w, mean, scale, repeat_count))
        std::cout << "ALL TEST DONE\n";

    return 0;
}
