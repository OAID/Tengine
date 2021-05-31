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
 * Copyright (c) 2018, OPEN AI LAB
 * Author: jingyou@openailab.com
 */
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <string.h>
#include "tengine_operations.h"
#include <iomanip>
#include <algorithm>
#include "tengine/c_api.h"
#include "common.hpp"

#define DEFAULT_MODEL_NAME "squeezenet"
#define DEFAULT_IMAGE_FILE "images/cat.jpg"
#define DEFAULT_LABEL_FILE "models/synset_words.txt"
#define DEFAULT_IMG_H 227
#define DEFAULT_IMG_W 227
#define DEFAULT_SCALE 1.f
#define DEFAULT_MEAN1 104.007
#define DEFAULT_MEAN2 116.669
#define DEFAULT_MEAN3 122.679
#define DEFAULT_REPEAT_CNT 1
#define DEFAULT_THREAD_CNT 1

#define MODEL_NUM  44
const Model_Config model_list[MODEL_NUM] = {
    {"squeezenet", 227, 227, 1.f, {104.007, 116.669, 122.679}, "", "squeezenet.tmfile", "synset_words.txt"},
    {"mobilenet", 224, 224, 0.017, {104.007, 116.669, 122.679}, "", "mobilenet.tmfile", "synset_words.txt"},
    {"mobilenet_v2", 224, 224, 0.017, {104.007, 116.669, 122.679}, "", "mobilenet_v2.tmfile", "synset_words.txt"},
    {"resnet50", 224, 224, 1.f, {104.007, 116.669, 122.679}, "", "resnet50.tmfile", "synset_words.txt"},
    {"alexnet", 227, 227, 1.f, {104.007, 116.669, 122.679}, "", "alexnet.tmfile", "synset_words.txt"},
    {"googlenet", 224, 224, 1.f, {104.007, 116.669, 122.679}, "", "googlenet.tmfile", "synset_words.txt"},
    {"inception_v3", 395, 395, 0.0078, {104.007, 116.669, 122.679}, "", "inception_v3.tmfile", "synset2015.txt"},
    {"inception_v4", 299, 299, 1 / 127.5f, {104.007, 116.669, 122.679}, "", "inception_v4.tmfile", "synset_words.txt"},
    {"vgg16", 224, 224, 1.f, {104.007, 116.669, 122.679}, "", "vgg16.tmfile", "synset_words.txt"},
    {"squeezenet_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "squeezenet_mx.tmfile", "synset_words.txt"},
    {"mobilenet_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "mobilenet_mx.tmfile", "synset_words.txt"},
    {"mobilenet_v2_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "mobilenet_v2_mx.tmfile", "synset_words.txt"},
    {"alexnet_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "alexnet_mx.tmfile", "synset_words.txt"},
    {"inception_v3_mx", 299, 299, 1.f, {0.485, 0.456, 0.406}, "", "inception_v3_mx.tmfile", "synset_words.txt"},
    {"resnet50_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "resnet50_mx.tmfile", "synset_words.txt"},
    {"vgg16_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "vgg16_mx.tmfile", "synset_words.txt"},
    {"squeezenet_on", 227, 227, 1.f, {104.007, 116.669, 122.679}, "", "squeezenet_on.tmfile", "synset_words.txt"},
    {"inception_v3_tflow", 299, 299, 0.0039, {0, 0, 0}, "", "inception_v3_tf.tmfile", "labels.txt"},
    {"inception_v4_tflow", 299, 299, 0.0039, {0, 0, 0}, "", "inception_v4_tf.tmfile", "labels.txt"},
    {"resnet_v2_tflow", 299, 299, 0.0039, {0, 0, 0}, "", "resnet_v2_tf.tmfile", "labels.txt"},
    {"mobilenet_tflow", 224, 224, 0.017, {104.007, 116.669, 122.679}, "", "mobilenet_v1_tf.tmfile", "labels.txt"},
    {"mobilenet_v2_tflow", 224, 224, 0.0078, {128, 128, 128}, "", "mobilenet_v2_tf.tmfile", "imagenet_slim_labels.txt"},
    {"squeezenet_tflow",224, 224, 0.0039, {0, 0, 0}, "", "squeezenet_tf.tmfile", "labels.txt"},
    {"resnet50_tflow", 224, 224, 1.f, {0, 0, 0}, "", "resnet50_tf.tmfile", "synset_words.txt"},
    {"inception_resnet_v2_tflow",299, 299, 0.0039, {0, 0, 0}, "","inception_resnet_v2_tf.tmfile","labels.txt"},
    {"mobilenet_v1_0_75_tflow", 224, 224, 0.017, {104.007, 116.669, 122.679}, "", "mobilenet_v1_0.75_tf.tmfile", "synset_words.txt"},
    {"mobilenet_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_quant_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"mobilenet_v2_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_v2_1.0_224_quant_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"inception_v3_tflite", 299, 299, 1.f, {0, 0, 0}, "", "inception_v3_quant_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"mobilenet_75_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_v1_quant_0.75_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"nasnet_tflow", 224, 224, 1/255.f, {0, 0, 0}, "", "nasnet_tf.tmfile", "synset_words.txt"},
    {"densenet_tflow", 224, 224, 0.0039, {128, 128, 128}, "", "densenet_tf.tmfile", "imagenet_slim_labels.txt"},
    {"mnasnet", 224, 224, 0.017, {104.007, 116.669, 122.679}, "", "mnasnet.tmfile", "synset_words.txt"},
    {"shufflenet_1xg3", 224, 224, 0.017, {103.940, 116.780, 123.680}, "", "shufflenet_1xg3.tmfile", "synset_words.txt"},
    {"shufflenet_v2", 224, 224, 1/255.f, {103.940, 116.780, 123.680}, "", "shufflenet_v2.tmfile", "synset_words.txt"},
    {"mobilenetv2_0_25_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "mobilenetv2_0_25_mx.tmfile", "synset_words.txt"},
    {"resnet18_v2_mx", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "resnet18_v2_mx.tmfile", "synset_words.txt"},
    {"mobilenet_0_25_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_0_25_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"mobilenet_0_5_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_0_5_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"mobilenet_0_75_tflite", 224, 224, 1.f, {0, 0, 0}, "", "mobilenet_0_75_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"inception_v4_tflite", 299, 299, 0.0039, {0, 0, 0}, "", "inception_v4_tflite.tmfile", "labels.txt"},
    {"squeezenet_tflite", 224, 224, 0.0039, {0, 0, 0}, "", "squeezenet_tflite.tmfile", "imagenet_slim_labels.txt"},
    {"mobilenet_v3_on", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "mobilenet_v3_on.tmfile", "synset_words.txt"},
    {"shufflenet_v2_on", 224, 224, 1.f, {0.485, 0.456, 0.406}, "", "shufflenet_v2_on.tmfile", "synset_words.txt"}
    };

void softmax(float* src, size_t length)
{
    float max = *std::max_element(src, src+length);
    float* buffer = (float*)malloc(length *sizeof(float));

    float sum = 0.0f;

    for (int i = 0; i < length; ++i)
    {
        buffer[i] = std::exp(src[i] - max);
    }

    for (int i = 0; i < length; ++i)
    {
        sum += buffer[i];
    }

    for (int i = 0; i < length; ++i)
    {
        src[i] = buffer[i] / sum;
    }

    free(buffer);
}

void get_input_data_mx(const char* image_file, float* input_data, int img_h, int img_w, const float* mean)
{
    float scale[3] = {0.229, 0.224, 0.225};   
    float means[3] = {mean[0], mean[1], mean[2]};
    image img = imread(image_file, img_w, img_h, means, scale, MXNET);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);
    free_image(img); 
}

void get_input_data_onnx(const char* image_file, float* input_data, int img_h, int img_w, const float* mean)
{
    float tmp_scale[3] = {1.0f, 1.0f, 1.0f}; 
    float tmp_means[3] = {0, 0, 0};
    // for some reaseon, wo use Tengine imread API to read image with CAFFE param, but do not normalize data.
    image img = imread(image_file, img_w, img_h, tmp_means, tmp_scale, CAFFE);
    img = rgb2bgr_premute(img);

    // normalize image data use mean & scale
    float scale_img[3] = {0.229, 0.224, 0.225}; 
    float means_img[3] = {0.485, 0.456, 0.406};  
    for(int c = 0; c < img.c; c++){
        for(int i = 0; i < img.h; i++){
            for(int j = 0; j < img.w; j++){
                int index = c*img.h*img.w + i * img.w + j;
                img.data[index] = ((img.data[index] / 255) - means_img[c]) / scale_img[c];
            }
        }
    }

    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);
    free_image(img); 
}

void LoadLabelFile_nasnet(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

bool run_tengine_library(const char* model_name, const char* tm_file, const char* label_file, const char* image_file,
                         int img_h, int img_w, const float* mean, float scale, int repeat_count,const std::string device, const int num_thread)
{
    // init
    init_tengine();
    std::cout << "tengine library version: " << get_tengine_version() << "\n";
    if(request_tengine_version("1.0") < 0)
        return false;

    // create graph
    graph_t graph = create_graph(nullptr, "tengine", tm_file);
    if(graph == nullptr)
    {
        std::cerr << "Create graph failed.\n";
        return false;
    }

    // set input shape
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};

    float* input_data = (float* )malloc(sizeof(float) * img_size);
    uint8_t* input_data_tflite = (uint8_t* )malloc(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::cerr << "Get input tensor failed\n";
        return false;
    }

    set_tensor_shape(input_tensor, dims, 4);
    // set the device to execution the graph
    if(!device.empty())
    {
        set_graph_device(graph, device.c_str());
    }

    /* prerun the graph */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    if(std::getenv("NumThreadLite"))
        opt.num_thread = atoi(std::getenv("NumThreadLite"));
    if(std::getenv("NumClusterLite"))
        opt.cluster = atoi(std::getenv("NumClusterLite"));
    if(std::getenv("DataPrecision"))
        opt.precision = atoi(std::getenv("DataPrecision"));
    if(std::getenv("REPEAT"))
        repeat_count = atoi(std::getenv("REPEAT"));
    
    std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
    std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
    std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";
    std::cout<<"Number Repeat  : [" << repeat_count <<"], use export REPEAT=10/100/1000 set\n";

    if(prerun_graph_multithread(graph, opt) < 0)
    {
        std::cout << "Prerun graph failed, errno: \n";
        return -1;
    }

    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    std::string str = model_name;
    std::string::size_type p1 = str.find("_tflow");
    std::string::size_type p2 = str.find("_mx");
    std::string::size_type p3 = str.find("_tflite");
    std::string::size_type p4 = str.find("_on");
    if(p1 != std::string::npos)
    {
        get_input_data_tf(image_file, input_data, img_h, img_w, mean, scale);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
    }
    else if(p2 != std::string::npos)
    {
        get_input_data_mx(image_file, input_data, img_h, img_w, mean);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
    }
    else if(p3 != std::string::npos)
    {
        if ((strstr(model_name, "inception_v4_tflite") != 0) ||
            (strstr(model_name, "squeezenet_tflite") != 0) ||
            (strstr(model_name, "mobilenet_v1_tflite") != 0) ||
            (strstr(model_name, "mobilenet_v2_tflite") != 0))
        {
            get_input_data_tf(image_file, input_data, img_h, img_w, mean, scale);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);
        }
        else
        {
            get_input_data_uint8(image_file, input_data_tflite, img_h, img_w);
            set_tensor_buffer(input_tensor, input_data_tflite, img_size * 1);
        }
    }
    else if(p4 != std::string::npos)
    {
        if((strstr(model_name, "mobilenet_v3_on") != 0) || (strstr(model_name, "shufflenet_v2_on") != 0))
        {
            get_input_data_onnx(image_file, input_data, img_h, img_w, mean);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);   
        }
        else
        {
            get_input_data(image_file, input_data, img_h, img_w, mean, scale);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);   
        }   
    }
    else
    {
        get_input_data(image_file, input_data, img_h, img_w, mean, scale);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);            
    }
    
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return false;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
        printf("current %.3f ms\n", mytime);
    }
    std::cout << "\nModel name : " << model_name << "\n"
              << "tengine model file : " << tm_file << "\n"
              << "label file : " << label_file << "\n"
              << "image file : " << image_file << "\n"
              << "img_h, imag_w, scale, mean[3] : " << img_h << " " << img_w << " " << scale << " " << mean[0] << " "
              << mean[1] << " " << mean[2] << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";
    std::cout << "--------------------------------------\n";

    // print output
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    if(strstr(model_name, "_tflite") != 0)
    {
        if ((strstr(model_name, "inception_v4_tflite") != 0) ||
            (strstr(model_name, "squeezenet_tflite") != 0) ||
            (strstr(model_name, "mobilenet_v1_tflite") != 0) ||
            (strstr(model_name, "mobilenet_v2_tflite") != 0))
        {
            float* data = ( float* )get_tensor_buffer(output_tensor);
            int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
            PrintTopLabels_common(label_file, data, data_size, model_name);
        }
        else
        {
            uint8_t* data = ( uint8_t* )get_tensor_buffer(output_tensor);
            int data_size = get_tensor_buffer_size(output_tensor);
            float output_scale = 0.0f;
            int zero_point = 0;
            get_tensor_quant_param(output_tensor, &output_scale, &zero_point, 1);
            PrintTopLabels_uint8(label_file, data, data_size, output_scale, zero_point);
        }
    }
    else if (strstr(model_name, "nasnet_tflow") != 0)
    {
        std::string output_tensor_name = "final_layer/predictions";
        output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());
        float* data = ( float* )get_tensor_buffer(output_tensor);
        int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
        float* end = data + data_size;

        std::vector<float> result(data, end);
        std::vector<int> top_N = Argmax(result, 5);
        std::vector<std::string> labels;

        LoadLabelFile_nasnet(labels, label_file);

        for(int i=0; i<top_N.size(); i++)
        {
            int idx = top_N[i];

            std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
            std::cout << labels[idx-1] << "\"\n";
        }        
    }
    else
    {
        float* data = (float* )get_tensor_buffer(output_tensor);
        int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
        if((strstr(model_name, "mobilenet_v3_on") != 0) || (strstr(model_name, "shufflenet_v2_on") != 0))
        {
            softmax(data, data_size);
        }
        PrintTopLabels_common(label_file, data, data_size, model_name);
    }
    std::cout << "--------------------------------------\n";

    
    // test output data fp32
    int out_data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
    
    std::string model_test = model_name;
    std::string out_file = "./data/" + model_test + "_out.bin";

    if (strstr(model_name, "nasnet_tflow") != 0)
    {
        std::string output_tensor_name = "final_layer/predictions";
        output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());
        out_data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
    }
    float* out_data = ( float* )get_tensor_buffer(output_tensor);
    
    float* out_data_ref = (float*)malloc(out_data_size * sizeof(float));
    FILE *fp;  
    fp=fopen(out_file.c_str(),"rb");
    if(fread(out_data_ref, sizeof(float), out_data_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);
    if(float_mismatch(out_data_ref, out_data, out_data_size) != true)
        return false;

    
    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();

    return true;
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_CNT;
    int num_thread = DEFAULT_THREAD_CNT;
    const std::string root_path = get_root_path();
    std::string model_name;
    std::string tm_file;
    std::string label_file;
    std::string image_file;
    std::string device;
    std::vector<int> hw;
    std::vector<float> ms;
    int img_h = 0;
    int img_w = 0;
    float scale = 0.0;
    float mean[3] = {-1.0, -1.0, -1.0};

    int res;
    while((res = getopt(argc, argv, "d:n:t:l:i:g:s:w:r:m:h")) != -1)
    {
        switch(res)
        {
            case 'd':
                device = optarg;
                break;
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
        mod_config = get_model_config(model_list, MODEL_NUM, model_name.c_str());
        if(mod_config == nullptr)
            return -1;

        // get tm file
        tm_file = get_file(mod_config->model_file);
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
                            mean, scale, repeat_count, device, num_thread))
        return -1;

    std::cout << "ALL TEST DONE\n";

    return 0;
}

