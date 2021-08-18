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
 * Author: xlchen@openailab.com
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "math.h"
#include "getopt.h"
#include "string.h"

#include "tengine/c_api.h"

static void split(float* array, char* str, const char* del)
{
    char* s = NULL;
    s = strtok(str, del);
    while (s != NULL)
    {
        *array++ = atof(s);
        s = strtok(NULL, del);
    }
}

int onnx_model_test(std::string model_file, int img_c, int img_h, int img_w)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 255;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    // fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file.c_str());
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the shape, data buffer of input_tensor of the graph */
    int img_size = img_h * img_w * img_c;
    int dims[] = {1, img_c, img_h, img_w}; // nchw
    float* input_data = (float*)malloc(img_size * sizeof(float));

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float)) < 0)
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
    for (size_t i = 0; i < img_size; i++)
    {
        input_data[i] = 1.f;
    }

    /* run graph */
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }

    int out_tensor_num = get_graph_output_node_number(graph);
    for (int tensor_id = 0; tensor_id < out_tensor_num; tensor_id++)
    {
        /* get the result of classification */
        tensor_t output_tensor = get_graph_output_tensor(graph, tensor_id, 0);
        float* output_data = (float*)get_tensor_buffer(output_tensor);
        int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
        const char* tensor_name = get_tensor_name(output_tensor);
        fprintf(stderr, "test output tensor: %s begin\n", tensor_name);

        size_t sub_start = model_file.find("/");
        size_t sub_end = model_file.find(".");
        std::string mode_name = model_file.substr(sub_start, sub_end - sub_start);
        std::string val_data = "./outputs" + mode_name + "_" + std::to_string(tensor_id) + ".txt";
        std::ifstream f(val_data);
        if (!f.is_open())
        {
            fprintf(stderr, "open val file %s failed.\n", val_data.c_str());
            return -1;
        }

        std::string line_str;
        char* end;
        int onnx_out_size = 1;
        while (std::getline(f, line_str))
        {
            // std::cout << line_str << std::endl;
            if (line_str == "shape:")
                continue;
            if (line_str == "data:")
                break;
            onnx_out_size *= strtol(line_str.c_str(), &end, 10);
        }
        // std::cout << onnx_out_size << std::endl;
        if (onnx_out_size != output_size)
        {
            fprintf(stderr, "not equal on size\n");
            fprintf(stderr, "tengine:%d,onnx:%d\n", output_size, onnx_out_size);
            fprintf(stderr, "test model:%s failed!\n", model_file.c_str());
            return -1;
        }

        float* onnx_out_data = (float*)malloc(sizeof(float) * onnx_out_size);
        int i = 0;
        while (std::getline(f, line_str))
        {
            std::stringstream ss(line_str);
            std::string str;
            int j = 0;
            while (getline(ss, str, ' '))
            {
                float tmp = strtof32(str.c_str(), &end);
                onnx_out_data[i++] = tmp;
            }
        }
        for (size_t i = 0; i < output_size; i++)
        {
            if (fabs(output_data[i] - onnx_out_data[i]) > 0.001)
            {
                fprintf(stderr, "not equal on data\n");
                fprintf(stderr, "tengine:%f,onnx:%f\n", output_data[i], onnx_out_data[i]);
                fprintf(stderr, "test model:%s failed!\n", model_file.c_str());
                return -1;
            }
        }
        fprintf(stderr, "test output tensor: %s done.\n", tensor_name);
    }

    fprintf(stderr, "test model: %s pass!\n", model_file.c_str());

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-g img_c,img_h,img_w] \n");
    fprintf(
        stderr,
        "\nexample: \n    ./examples/onnx_ci -m ssd-sim.tmfile -g 3,300,300\n");
}

int main(int argc, char* argv[])
{
    char* model_file = NULL;
    char* image_file = NULL;
    float img_hw[3] = {0.f};
    int img_c = 0;
    int img_h = 0;
    int img_w = 0;

    int res;
    while ((res = getopt(argc, argv, "m:g:h")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'g':
            split(img_hw, optarg, ",");
            img_c = (int)img_hw[0];
            img_h = (int)img_hw[1];
            img_w = (int)img_hw[2];
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == NULL || img_c == 0 || img_h == 0 || img_w == 0)
    {
        show_usage();
        return -1;
    }

    if (onnx_model_test(model_file, img_c, img_h, img_w) < 0)
        return -1;

    return 0;
}
