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
#include <functional>
#include <algorithm>

#include "common.hpp"
#include <sys/time.h>
#include "tengine_c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define DEF_MODEL "models/landmark.tmfile"
#define DEF_IMAGE "images/mobileface02.jpg"

// void get_input_fp32_data(const char* image_file, float* input_data, int img_h, int img_w, float* mean, float* scale)
// {
//     image img = imread_process(image_file, img_w, img_h, mean, scale);

//     float* image_data = ( float* )img.data;

//     for (int i = 0; i < img_w * img_h * 3; i++)
//         input_data[i] = image_data[i];

//     free_image(img);
// }

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    const std::string root_path = get_root_path();
    std::string model_file;
    std::string image_file;
    int img_h = 144;
    int img_w = 144;
    float mean[3] = {128.f, 128.f, 128.f};
    // float scale[3] = {0.0039, 0.0039, 0.0039};
    float scale = 0.0039;

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

    // load model
    if(model_file.empty())
    {
        model_file = root_path + DEF_MODEL;
        std::cout << "model file not specified,using " << model_file << " by default\n";
    }
    if(image_file.empty())
    {
        image_file = root_path + DEF_IMAGE;
        std::cout << "image file not specified,using " << image_file << " by default\n";
    }
    // check file
    if((!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
        return -1;
    }



    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file.c_str());
    if (graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
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
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    // get_input_fp32_data(image_file, input_data, img_h, img_w, mean, scale);
    get_input_data(image_file.c_str(), input_data, img_h, img_w, mean, scale);

    /* run graph */
    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for (int i = 0; i < repeat_count; i++)
    {
        // double start = get_current_time();
        gettimeofday(&t0, NULL);
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
           avg_time / repeat_count);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

    image img_out = imread(image_file.c_str());
    for (int i = 0; i < data_size / 2; i++)
    {
        int x = ( int )(data[2 * i] * ( float )img_out.w / 144.f);
        int y = ( int )(data[2 * i + 1] * ( float )img_out.h / 144.f);
        draw_circle(img_out, x, y, 2, 0, 255, 0);
    }

    save_image(img_out, "landmarkout");

    // // save output data
    // FILE *outfp;  
    // outfp=fopen("/opt/data/landmark_out.bin", "wb");
    // fwrite(data, sizeof(float), data_size, outfp);
    // fclose(outfp);
    // test output data
    float* buf = ( float* )malloc(data_size * sizeof(float));
    FILE *fp;  
    fp=fopen("./data/landmark_out.bin","rb");
    if(fread(buf, sizeof(float), data_size, fp)==0)
    {
        printf("read ref data file failed!\n");
        return false;
    }
    fclose(fp);
    
    if(float_mismatch(buf, data, data_size) != true)
        return -1;

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
