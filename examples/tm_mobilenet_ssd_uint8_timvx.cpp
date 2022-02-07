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

#include <iostream>
#include <string>
#include <vector>
#include "tengine_operations.h"
#include "tengine/c_api.h"
#include "common.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

typedef struct Box
{
    int x0;
    int y0;
    int x1;
    int y1;
    int class_idx;
    float score;
} Box_t;

void get_input_uint_data_ssd(const char* image_file, uint8_t* input_data, int img_h, int img_w, float input_scale,
                             int zero_point)
{
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float scales[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    image img = imread_process(image_file, img_w, img_h, mean, scales);

    float* image_data = (float*)img.data;

    for (int i = 0; i < img_w * img_h * 3; i++)
    {
        int udata = round(image_data[i] / input_scale + zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;

        input_data[i] = udata;
    }

    free_image(img);
}

void post_process_ssd(const char* image_file, float threshold, float* outdata, int num)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                                 "bus", "car", "cat", "chair", "cow", "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                 "sofa", "train", "tvmonitor"};

    image im = imread(image_file);

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
            printf("BOX:( %d , %d ),( %d , %d )\n", (int)box.x0, (int)box.y0, (int)box.x1, (int)box.y1);
        }
        outdata += 6;
    }
    for (int i = 0; i < (int)boxes.size(); i++)
    {
        Box box = boxes[i];
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    save_image(im, "mobilenet_ssd_uint8_timvx_out");
    free_image(im);
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t"
              << "Mobilenet_SSD"
              << "\n";
    std::cout << "======================================\n";
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;
    int img_h = 300;
    int img_w = 300;
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float scale[3] = {0.007843f, 0.007843f, 0.007843f};
    float show_threshold = 0.5f;
    int ret;
    while ((ret = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (ret)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == NULL)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    // init tengine
    if (init_tengine() < 0)
    {
        std::cout << "init tengine failed\n";
        return 1;
    }

    // create VeriSilicon TIM-VX backend
    context_t timvx_context = create_context("timvx", 1);
    if (set_context_device(timvx_context, "TIMVX", nullptr, 0) < 0)
    {
        fprintf(stderr, "add_context_device failed.\n");
        return 1;
    }

    // create graph
    //graph_t graph = create_graph(nullptr, "tengine", model_file);
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        return 1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    uint8_t* input_data = (uint8_t*)malloc(img_size * sizeof(uint8_t));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(uint8_t)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    get_input_uint_data_ssd(image_file, input_data, img_h, img_w, input_scale, input_zero_point);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", 1, 1,
            total_time / 1, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0); //"detection_out"
    int out_dim[4];
    get_tensor_shape(output_tensor, out_dim, 4);
    int output_size = get_tensor_buffer_size(output_tensor);
    uint8_t* output_u8 = (uint8_t*)get_tensor_buffer(output_tensor);
    float* output_data = (float*)malloc(output_size * sizeof(float));

    /* dequant */
    float output_scale = 0.f;
    int output_zero_point = 0;
    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
    for (int i = 0; i < output_size; i++)
        output_data[i] = ((float)output_u8[i] - (float)output_zero_point) * output_scale;

    /* post_process_ssd */
    post_process_ssd(image_file, show_threshold, output_data, out_dim[1]);

    /* release tengine */
    free(output_data);
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
