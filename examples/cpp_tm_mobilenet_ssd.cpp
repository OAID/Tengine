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

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>

#include "common.h"
#include "tengine_cpp_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

using namespace std;

typedef struct Box
{
    int x0;
    int y0;
    int x1;
    int y1;
    int class_idx;
    float score;
} Box_t;

void post_process_ssd(const string image_file, float threshold, const float* outdata, int num)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                                 "bus", "car", "cat", "chair", "cow", "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                 "sofa", "train", "tvmonitor"};

    image im = imread(image_file.c_str());

    int raw_h = im.h;
    int raw_w = im.w;
    //    struct vector* boxes = create_vector(sizeof(Box_t), nullptr);
    std::vector<Box_t> boxes;

    fprintf(stderr, "detect result num: %d \n", num);
    for (int i = 0; i < num; i++)
    {
        if (outdata[1] >= threshold)
        {
            Box_t box;

            box.class_idx = (int)outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;

            boxes.push_back(box);
            fprintf(stderr, "%s\t:%.1f%%\n", class_names[box.class_idx], box.score * 100);
            fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }
    for (int i = 0; i < boxes.size(); i++)
    {
        Box_t box = boxes[i];
        draw_box(im, box.x0, box.y0, box.x1, box.y1, 2, 125, 0, 125);
    }

    save_image(im, "mobilenet_ssd_out");
    free_image(im);
    fprintf(stderr, "======================================\n");
    fprintf(stderr, "[DETECTED IMAGE SAVED]:\n");
    fprintf(stderr, "======================================\n");
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    string model_file;
    string image_file;
    int img_h = 300;
    int img_w = 300;
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float scale[3] = {0.007843f, 0.007843f, 0.007843f};
    float show_threshold = 0.5f;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
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
    if (model_file.empty())
    {
        std::cerr << "Error: Tengine model file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    if (image_file.empty())
    {
        std::cerr << "Error: Image file not specified!" << std::endl;
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file.c_str()) || !check_file_exist(image_file.c_str()))
        return -1;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* net inference */
    {
        tengine::Net somenet;
        tengine::Tensor input_tensor;
        tengine::Tensor output_tensor;

        /* set runtime options of Net */
        somenet.opt.num_thread = num_thread;
        somenet.opt.cluster = TENGINE_CLUSTER_ALL;
        somenet.opt.precision = TENGINE_MODE_FP32;

        /* load model */
        somenet.load_model(nullptr, "tengine", model_file.c_str());

        /* prepare input data */
        input_tensor.create(1, 3, img_h, img_w);
        get_input_data(image_file.c_str(), (float*)input_tensor.data, img_h, img_w, mean, scale);

        /* forward */
        somenet.input_tensor("data", input_tensor);

        double min_time, max_time, total_time;
        min_time = DBL_MAX;
        max_time = DBL_MIN;
        total_time = 0;
        for (int i = 0; i < repeat_count; i++)
        {
            double start_time = get_current_time();
            somenet.run();
            double end_time = get_current_time();
            double cur_time = end_time - start_time;

            total_time += cur_time;
            max_time = std::max(max_time, cur_time);
            min_time = std::min(min_time, cur_time);
        }
        printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
               total_time / repeat_count);

        /* get result */
        somenet.extract_tensor("detection_out", output_tensor);

        /* SSD process */
        post_process_ssd(image_file, show_threshold, (float*)output_tensor.data, output_tensor.h);
    }

    /* release */
    release_tengine();

    return 0;
}
