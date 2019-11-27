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
 * Author: chunyinglv@openailab.com
 */

#include "cpu_device.h"
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include <fstream>
#include <iomanip>    //std::setprecision(4)
#include <iostream>
#include <sys/time.h>
#include <thread>
#include <vector>

const char* label_file = "./models/synset_words.txt";
const char* proto1 = "./models/sqz.prototxt";
const char* model1 = "./models/squeezenet_v1.1.caffemodel";
const char* mname1 = "sqz";
const char* img_name1 = "./tests/images/cat.jpg";
const float channel_mean[3] = {104.007, 116.669, 122.679};

const char* proto2 = "models/MobileNetSSD_deploy.prototxt";
const char* model2 = "models/MobileNetSSD_deploy.caffemodel";
const char* mname2 = "mssd";
const char* img_name2 = "tests/images/ssd_dog.jpg";

int cpu_2A72_repeat_count = 10;
int cpu_4A53_repeat_count = 10;

static inline bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

static inline std::vector<int> Argmax(const std::vector<float>& v, int N)
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
    std::vector<int> top_N = Argmax(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};
void get_input_data_sqz(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    image im = imread(image_file);
    image imRes;
    if(im.c == 1)
    {
        imRes = gray2bgr(im);
    }
    else
    {
        imRes = im;
    }
    image imResize = resize_image(imRes, img_w, img_h);
    imResize = rgb2bgr_premute(imResize);
    int hw = img_w * img_h;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (imResize.data[c * hw + h * img_w + w] - mean[c]) * scale;
            }
}

void get_input_data_ssd(const char* image_file, float* input_data, int img_h, int img_w)
{
    image im = imread(image_file);

    float mean[3] = {127.5, 127.5, 127.5};
    int hw = img_w * img_h;
    image imRes = resize_image(im, img_w, img_h);
    imRes = rgb2bgr_premute(imRes);
    for(int c = 0; c < 3; c++)
    {
        for(int h = 0; h < img_h; h++)
        {
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (imRes.data[c * hw + h * img_w + w] - mean[c]);
            }
        }
    }
}

void post_process_ssd(const char* image_file, float threshold, float* outdata, int num, const char* save_name)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};

    image im = imread(image_file);

    int raw_h = im.h;
    int raw_w = im.w;
    std::vector<Box> boxes;
    printf("detect result num: %d \n", num);
    for(int i = 0; i < num; i++)
    {
        if(outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
            printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }
    for(int i = 0; i < ( int )boxes.size(); i++)
    {
        Box box = boxes[i];

        std::ostringstream score_str;
        score_str << box.score * 100;
        std::string labelstr = std::string(class_names[box.class_idx]) + " : " + score_str.str();

        put_label(im, labelstr.c_str(), 0.02, box.x0, box.y0, 255, 255, 125);
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    save_image(im, save_name);

    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t" << save_name << "\n";
    std::cout << "======================================\n";
}

void run_mssd(graph_t graph, int repeat_count, const char* image_file)
{
    const char* save_name = "mssd_result.jpg";
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);

    if(input_tensor == nullptr)
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    prerun_graph(graph);

    struct timeval t0, t1;
    float total_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        get_input_data_ssd(image_file, input_data, img_h, img_w);

        gettimeofday(&t0, NULL);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }

    std::cout << "--------------------------------------\n";
    std::cout << "MSSD repeat " << repeat_count << " times, avg " << total_time / repeat_count
              << " ms  all: " << total_time << "ms\n";
    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    float show_threshold = 0.5;

    post_process_ssd(image_file, show_threshold, outdata, num, save_name);

    release_graph_tensor(out_tensor);
    release_graph_tensor(input_tensor);

    postrun_graph(graph);
    free(input_data);
}

void run_sqz(graph_t graph, int repeat_count, const char* image_file)
{
    int img_h = 227;
    int img_w = 227;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);

    if(input_tensor == nullptr)
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    prerun_graph(graph);

    const float channel_mean[3] = {104.007, 116.669, 122.679};
    struct timeval t0, t1;
    float total_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        get_input_data_sqz(image_file, input_data, img_h, img_w, channel_mean, 1);
        gettimeofday(&t0, NULL);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }

    std::cout << "--------------------------------------\n";
    std::cout << "SQZ repeat " << repeat_count << " times, avg " << total_time / repeat_count
              << " ms  all: " << total_time << "ms\n";

    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");

    float* out_data = ( float* )(get_tensor_buffer(out_tensor));
    PrintTopLabels(label_file, out_data);

    release_graph_tensor(out_tensor);
    release_graph_tensor(input_tensor);

    postrun_graph(graph);
    free(input_data);
}

void thread2_mssd_4a53(const char* proto, const char* model, const char* model_name, const char* image_file)
{
    // create graph
    graph_t graph = create_graph(nullptr, "caffe", proto, model);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!\n";
        return;
    }

    std::cout << "load graph done!\n";

    // set device

    if(set_graph_device(graph, "a53") < 0)
    {
        std::cerr << "set device a53 failed\n";
    }
    run_mssd(graph, cpu_4A53_repeat_count, image_file);

    destroy_graph(graph);
}

void thread1_sqz_2a72(const char* proto, const char* model, const char* model_name, const char* image_file)
{
    // create graph
    graph_t graph = create_graph(nullptr, "caffe", proto, model);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!\n";
        return;
    }

    std::cout << "load graph done!\n";

    // set device
    if(set_graph_device(graph, "a72") < 0)
    {
        std::cerr << "set device a72 failed\n";
    }

    run_sqz(graph, cpu_2A72_repeat_count, image_file);

    destroy_graph(graph);
}

int main(int argc, char* argv[])
{
    // init tengine
    if(init_tengine() < 0)
    {
        std::cerr << "tengine init failed\n";
        return 1;
    }

    if(request_tengine_version("0.9") < 0)
    {
        std::cerr << "tengine run-time does not support 0.9\n";
        return 1;
    }

    // set device
    const struct cpu_info* p_info = get_predefined_cpu("rk3399");
    int a72_list[] = {4, 5};
    set_online_cpu(( struct cpu_info* )p_info, a72_list, sizeof(a72_list) / sizeof(int));
    create_cpu_device("a72", p_info);

    int a53_list[] = {0, 1, 2, 3};
    set_online_cpu(( struct cpu_info* )p_info, a53_list, sizeof(a53_list) / sizeof(int));
    create_cpu_device("a53", p_info);

    // thread1  run sqz (2A72)
    std::cout << "Thread 1 running sqz using 2A72 \n-------------------\n";
    std::thread* t1 = new std::thread(thread1_sqz_2a72, proto1, model1, mname1, img_name1);

    // thread2  run mssd (4A53)
    std::cout << "Thread 2 running mssd using 4A53 \n-------------------\n";
    std::thread* t2 = new std::thread(thread2_mssd_4a53, proto2, model2, mname2, img_name2);

    t1->join();
    t2->join();

    delete t1;
    delete t2;

    release_tengine();

    return 0;
}
