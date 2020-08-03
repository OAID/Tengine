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

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "common.h"

#define VXDEVICE "VX"

#define DEF_MODEL "models/mssd_caffe.tmfile"
#define DEF_IMAGE "images/ssd_dog.jpg"

typedef struct Box
{
    int x0;
    int y0;
    int x1;
    int y1;
    int class_idx;
    float score;
} Box_t;

void get_input_uint_data_ssd(std::string& image_file, uint8_t* input_data, int img_h, int img_w)
{
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    image img = imread_process(image_file.c_str(), img_w, img_h, mean, scales);

    float* image_data = ( float* )img.data;

    for (int i = 0; i < img_w * img_h * 3; i++)
    {
        int udata = int(image_data[i] / 0.009504f + 133);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;

        input_data[i] = udata;
    }

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
    for (int i = 0; i < num; i++)
    {
        if (outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = round(outdata[0]);
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
            printf("%s\t:%.2f\n", class_names[box.class_idx], box.score * 100.f);
            printf("BOX:( %d , %d ),( %d , %d )\n", ( int )box.x0, ( int )box.y0, ( int )box.x1, ( int )box.y1);
        }
        outdata += 6;
    }
    for (int i = 0; i < ( int )boxes.size(); i++)
    {
        Box box = boxes[i];

//        std::ostringstream score_str;
//        score_str << box.score * 100;
//        std::string labelstr = std::string(class_names[box.class_idx]) + " : " + score_str.str();

//        put_label(im, labelstr.c_str(), 0.02, box.x0, box.y0, 255, 255, 125);
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

int main(int argc, char* argv[])
{
    int ret = -1;
    std::string model_file, model_post_file;
    std::string image_file;
    std::string save_name = "save.jpg";
    float show_threshold = 0.5f;

    if (model_file.empty())
    {
        model_file = DEF_MODEL;
        std::cout << "model file not specified,using " << model_file << " by default\n";
    }

    if (image_file.empty())
    {
        image_file = DEF_IMAGE;
        std::cout << "image file not specified,using " << image_file << " by default\n";
    }

    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    uint8_t* input_data = ( uint8_t* )malloc(img_size);

    // init tengine
    if (init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }

    if (request_tengine_version("0.9") < 0)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }

    // create graph
    graph_t graph = create_graph(nullptr, "tengine", model_file.c_str());
    if (graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    dump_graph(graph);

    //set_graph_device(graph, VXDEVICE);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if (input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    ret = prerun_graph(graph);
    if (ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // set graph tensor
    get_input_uint_data_ssd(image_file, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size);

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < 1; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", 1,
            1, total_time / 1, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);    //"detection_out"
    int out_dim[4];
    get_tensor_shape(output_tensor, out_dim, 4);
    int output_size = get_tensor_buffer_size(output_tensor);
    uint8_t* output_u8 = (uint8_t*)get_tensor_buffer(output_tensor);
    float* output_data = (float*)malloc(output_size*sizeof(float));

    /* dequant */
    for (int i=0; i<output_size; i++)
        output_data[i] = ((float)output_u8[i] - 0.f) * 0.078698f;

    post_process_ssd(image_file, show_threshold, output_data, out_dim[1]);

    /* release tengine */
    free(output_data);
    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
