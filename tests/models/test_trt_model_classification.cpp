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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H        224
#define DEFAULT_IMG_W        224
#define DEFAULT_SCALE1       1.f
#define DEFAULT_SCALE2       1.f
#define DEFAULT_SCALE3       1.f
#define DEFAULT_MEAN1        104.007
#define DEFAULT_MEAN2        116.669
#define DEFAULT_MEAN3        122.679
#define DEFAULT_LOOP_COUNT   1
#define DEFAULT_THREAD_COUNT 1
#define DEFAULT_CPU_AFFINITY 255

int float_mismatch(float* current, float* reference, int size)
{
    int ret = 0;
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        if (fabs(tmp) > 0.001)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            ret = -1;
        }
    }
    fprintf(stderr, "test pass\n");

    return ret;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file]\n [-g img_h,img_w] [-s scale[0],scale[1],scale[2]] [-w "
        "mean[0],mean[1],mean[2]] [-r loop_count] [-t thread_count] [-a cpu_affinity]\n");
    fprintf(
        stderr,
        "\nmobilenet example: \n    ./classification -m /path/to/mobilenet.tmfile -i /path/to/img.jpg -g 224,224 -s "
        "0.017,0.017,0.017 -w 104.007,116.669,122.679\n");
}

int main(int argc, char* argv[])
{
    int loop_count = DEFAULT_LOOP_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    int cpu_affinity = DEFAULT_CPU_AFFINITY;
    std::string model_name;
    std::string model_file;
    char* image_file = NULL;
    float img_hw[2] = {0.f};
    int img_h = 0;
    int img_w = 0;
    float mean[3] = {-1.f, -1.f, -1.f};
    float scale[3] = {0.f, 0.f, 0.f};

    int res;
    while ((res = getopt(argc, argv, "m:i:l:g:s:w:r:t:a:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_name = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'g':
            split(img_hw, optarg, ",");
            img_h = (int)img_hw[0];
            img_w = (int)img_hw[1];
            break;
        case 's':
            split(scale, optarg, ",");
            break;
        case 'w':
            split(mean, optarg, ",");
            break;
        case 'r':
            loop_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'a':
            cpu_affinity = atoi(optarg);
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    model_file = "./models/" + model_name + ".tmfile";

    /* check files */
    if (model_file.empty())
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

    if (!check_file_exist(model_file.c_str()) || !check_file_exist(image_file))
        return -1;

    if (img_h == 0)
    {
        img_h = DEFAULT_IMG_H;
        fprintf(stderr, "Image height not specified, use default %d\n", img_h);
    }

    if (img_w == 0)
    {
        img_w = DEFAULT_IMG_W;
        fprintf(stderr, "Image width not specified, use default  %d\n", img_w);
    }

    if (scale[0] == 0.f || scale[1] == 0.f || scale[2] == 0.f)
    {
        scale[0] = DEFAULT_SCALE1;
        scale[1] = DEFAULT_SCALE2;
        scale[2] = DEFAULT_SCALE3;
        fprintf(stderr, "Scale value not specified, use default  %.1f, %.1f, %.1f\n", scale[0], scale[1], scale[2]);
    }

    if (mean[0] == -1.0 || mean[1] == -1.0 || mean[2] == -1.0)
    {
        mean[0] = DEFAULT_MEAN1;
        mean[1] = DEFAULT_MEAN2;
        mean[2] = DEFAULT_MEAN3;
        fprintf(stderr, "Mean value not specified, use default   %.1f, %.1f, %.1f\n", mean[0], mean[1], mean[2]);
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = cpu_affinity;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create NVIDIA TensorRT backend */
    context_t trt_context = create_context("trt", 1);
    int rtt = add_context_device(trt_context, "TensorRT");
    if (0 > rtt)
    {
        fprintf(stderr, "add_context_device NV TensorRT DEVICE failed.\n");
        return -1;
    }

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(trt_context, "tengine", model_file.c_str());
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    std::vector<float> input_data(img_size);

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

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data(image_file, input_data.data(), img_h, img_w, mean, scale);

    /* run graph */
    double start = get_current_time();
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }
    double end = get_current_time();

    fprintf(stderr, "\nmodel file : %s\n", model_file.c_str());
    fprintf(stderr, "image file : %s\n", image_file);
    fprintf(stderr, "img_h, img_w, scale[3], mean[3] : %d %d , %.3f %.3f %.3f, %.1f %.1f %.1f\n", img_h, img_w,
            scale[0], scale[1], scale[2], mean[0], mean[1], mean[2]);
    fprintf(stderr, "Inference time %.2f ms\n", end - start);
    fprintf(stderr, "--------------------------------------\n");

    /* get the result of classification */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* output_data = (float*)get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    print_topk(output_data, output_size, 5);
    fprintf(stderr, "--------------------------------------\n");

    /* check the result */
    std::string reference_file = "./data/" + model_name + "_out.bin";
    std::vector<float> reference_data(output_size);
    FILE* fp;
    fp = fopen(reference_file.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "read reference %s failed!\n", reference_file.c_str());
        return -1;
    }
    if (fread(reference_data.data(), sizeof(float), output_size, fp) == 0)
    {
        fprintf(stderr, "read reference data file failed!\n");
        return -1;
    }
    fclose(fp);

    int ret = float_mismatch(output_data, reference_data.data(), output_size);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret;
}
