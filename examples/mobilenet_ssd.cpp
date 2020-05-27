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
 * Author: sqfu@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include "tengine_operations.h"
#include "tengine_cpp_api.h"
#include "common.hpp"

static std::string gExcName{""};

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

void get_input_data_ssd(std::string& image_file, float* input_data, int img_h, int img_w)
{
    float mean[3] = {127.5, 127.5, 127.5};
    float scales[3] = {0.007843, 0.007843, 0.007843};
    image img = imread(image_file.c_str(), img_w, img_h, mean, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h);  
    free_image(img);    
}

void post_process_ssd(std::string& image_file, float threshold, float* outdata, int num)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};

    image im = imread(image_file.c_str());

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
            printf("BOX:( %d , %d ),( %d , %d )\n", (int)box.x0, (int)box.y0, (int)box.x1, (int)box.y1);
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

    save_image(im, "tengine_example_out");
    free_image(im);
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t"
              << "Mobilenet_SSD"
              << "\n";
    std::cout << "======================================\n";
}

void show_usage()
{
    std::cout << "[Usage]: " << gExcName << " [-h]\n"
              << "   [-m model_file] [-i image_file]\n";
}

int main(int argc, char* argv[])
{
    gExcName = std::string(argv[0]);
    int ret = -1;
    std::string model_file;
    std::string image_file;
    const char* device = nullptr;

    int res;
    while((res = getopt(argc, argv, "m:i:hd:")) != -1)
    {
        switch(res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'd':
                device = optarg;
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    if( model_file.empty() )
    {
        std::cerr << "Error: model file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    if(image_file.empty())
    {
        std::cerr << "Error: image file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    tengine::Net somenet;
    tengine::Tensor input_tensor;
    tengine::Tensor output_tensor;
	
    if(request_tengine_version("0.9") != 1)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }
    // check file
    if((!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
        return 1;
    }

    /* load model */
    somenet.load_model(NULL, "tengine", model_file.c_str());

    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;

    float mean[3] = {127.5, 127.5, 127.5};

    float scales = 0.007843;

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    /* prepare input data */
    input_tensor.create(img_w, img_h, 3);

    get_input_data(image_file.c_str(), (float* )input_tensor.data, img_h, img_w, mean, scales);

    /* forward */
    somenet.input_tensor(0, 0, input_tensor);

    struct timeval t0, t1;
    float total_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        somenet.run();
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "--------------------------------------\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";

    //tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
    /* get result */
    somenet.extract_tensor(0, 0, output_tensor);

    float* outdata = ( float* )(output_tensor.data);

    float show_threshold = 0.5;

    post_process_ssd(image_file, show_threshold, outdata, output_tensor.h);

    return 0;
}

