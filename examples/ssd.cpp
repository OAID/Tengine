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

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "tengine_operations.h"
#include "tengine/c_api.h"
#include <sys/time.h>
#include "common.h"

#define DEF_MODEL "models/ssd.tmfile"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

template<typename T>
void tengine_resize(T* input, float* output, int img_w, int img_h, int c, int h, int w){
    if(sizeof(T) == sizeof(float))
    	tengine_resize_f32((float*)input, output, img_w, img_h, c, h, w);
    // if(sizeof(T) == sizeof(uint8_t))
	//     tengine_resize_uint8((uint8_t*)input, output, img_w, img_h, c, h, w);
}

image rgb2bgr_premute(image src)
{
    float* GRB = ( float* )malloc(sizeof(float) * src.c * src.h * src.w);
    for(int c = 0; c < src.c; c++)
    {
        for(int h = 0; h < src.h; h++)
        {
            for(int w = 0; w < src.w; w++)
            {
                int newIndex = ( c )*src.h * src.w + h * src.w + w;
                int grbIndex = (2 - c) * src.h * src.w + h * src.w + w;
                GRB[grbIndex] = src.data[newIndex];
            }
        }
    }
    for(int c = 0; c < src.c; c++)
    {
        for(int h = 0; h < src.h; h++)
        {
            for(int w = 0; w < src.w; w++)
            {
                int newIndex = ( c )*src.h * src.w + h * src.w + w;
                src.data[newIndex] = GRB[newIndex];
            }
        }
    }
    free(GRB);
    return src;
}

image imread(const char* filename, int img_w, int img_h, float* means, float* scale, FUNCSTYLE func){

    image out = imread(filename);
    //image resImg = resize_image(out, img_w, img_h);
    image resImg = make_image(img_w, img_h, out.c);


    int choice = 0;
    if(out.c == 1){
        choice = 0;
    } else {
        choice = 2;
    }
    switch(choice){
        case 0:
            out = gray2bgr(out);
            break;
        case 1:
            out = rgb2gray(out);
            break;
        case 2:
            if(func != 2)
                out = rgb2bgr_premute(out);
            break;
        default:
            break;
    }

    switch(func){
        case 0:
            tengine_resize(out.data, resImg.data, out.w, out.h, out.c, out.h, out.w);
            free_image(out);
            return resImg;
            break;
        case 1:
            tengine_resize(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
            resImg = imread2caffe(resImg, img_w, img_h,   means,  scale);
            break;
        // case 2: 
        //     tengine_resize(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
        //     #ifdef CONFIG_LITE_TEST
        //     resImg = imread2caffe(resImg, img_w, img_h,   means,  scale);
        //     #else
        //     resImg = imread2tf(resImg,   img_w,   img_h,  means, scale);
        //     #endif
        //     break;
        // case 3:
        //     tengine_resize(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
        //     resImg = imread2mxnet( resImg,  img_w,  img_h,  means,  scale);
        //     break;
        // case 4:
        //     tengine_resize(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
        //     resImg = imread2tflite( resImg,  img_w,  img_h,  means,  scale);
        default:
            break;
    }
    free_image(out);
    return resImg;
}

void get_input_data_ssd(const char* image_file, float* input_data, int img_h, int img_w)
{
    float mean[3] = {127.5, 127.5, 127.5};
    float scales[3] = {1, 1, 1};
    image img = imread(image_file, img_w, img_h, mean, scales, CAFFE);    
    memcpy(input_data, img.data, sizeof(float)*3*img_w*img_h); 
    free_image(img);
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

        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    save_image(im, "SSD.jpg");
    free_image(im);
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t"
              << "SSD"
              << "\n";
    std::cout << "======================================\n";
}

int main(int argc, char* argv[])
{
    int ret = -1;
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* save_name = "ssd.jpg";

    int res;
    while((res = getopt(argc, argv, "m:i:")) != -1)
    {
        switch(res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        return -1;
    }
    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    // init tengine
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    // create graph
    graph_t graph = create_graph(nullptr, "tengine", model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    ret = prerun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float total_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        get_input_data_ssd(image_file, input_data, img_h, img_w);

        gettimeofday(&t0, NULL);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        ret = run_graph(graph, 1);
        if(ret != 0)
        {
            std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }
    std::cout << "--------------------------------------\n";
    std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";
    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out");
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    float show_threshold = 0.5;

    post_process_ssd(image_file, show_threshold, outdata, num, save_name);

    release_graph_tensor(input_tensor);
    release_graph_tensor(out_tensor);
    ret = postrun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    free(input_data);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
