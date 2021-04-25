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
 * Copyright (c) 2020, Open AI Lab
 * Author: sjliu@openailab.com
 */

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <vector>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "common.hpp"
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>

#define DEF_MODEL "models/mfn_14Mb_v2.0_fp32_20190819.tmfile"
#define DEF_IMAGE "images/male00000281_test.jpg"
#define DEF_IMAGE2 "images/male00000281_reg.jpg"
#define DEF_IMAGE3 "images/jiayuan_149119121d.jpg"
#define DEFAULT_IMG_H 112
#define DEFAULT_IMG_W 112
#define DEFAULT_REPEAT_CNT 1
#define FEATURESIZE 512

/* calculate cosine distance of two vectors */
float cosine_dist(float* vectorA, float* vectorB, int size)
{
    float Numerator=0;
    float Denominator1=0;
    float Denominator2=0;
    float Similarity;
    for (int i = 0 ; i < size ; i++)
    {
        Numerator += (vectorA[i] * vectorB[i]);
        Denominator1 += (vectorA[i] * vectorA[i]);
        Denominator2 += (vectorB[i] * vectorB[i]);
    }

    Similarity = Numerator/sqrt(Denominator1)/sqrt(Denominator2);

    return Similarity;
}

int run_tengine_library(const char* tm_file, std::string& image_file, int img_h, int img_w, int repeat_count, const std::string device, float *feature)
{
	printf("start create graph\n");
    // create graph
    graph_t graph = create_graph(nullptr, "tengine", tm_file);

    printf("create graph done\n");

    if(graph == nullptr)
    {
        std::cerr << "Create graph failed.\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return false;
    }

    // set input shape
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    image img = imread(image_file.c_str());
    // image img = resize_image(im, img_w, img_h);

    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    memcpy(input_data, img.data, sizeof(float)*img_size);

    // free_image(im);
    free_image(img);
    
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if(input_tensor == nullptr)
    {
        std::cerr << "Get input tensor failed\n";
        return false;
    }
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);

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
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    //dump_graph(graph);
    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;

    if (repeat_count > 1)
    {
        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);
        run_graph(graph, 1);   
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
    }
    std::cout << "tengine model file : " << tm_file << "\n"
              << "image file : " << image_file << "\n"
              << "img_h, imag_w : " << img_h << " " << img_w << "\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";
    std::cout << "--------------------------------------\n";

    /* convert output tensor name into a tensor handle */
    tensor_t output_tensor = get_graph_tensor(graph, "fc1");

    /* get the buffer of output tensor */
    float *data = (float *)get_tensor_buffer(output_tensor);
    memcpy(feature, data, FEATURESIZE * sizeof(float));

    std::cout << "--------------------------------------\n";

    free(input_data);
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    return 1;
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_CNT;
    const std::string root_path = get_root_path();
    std::string tm_file;
    std::string image_file;
    std::string image_file2;
    std::string image_file3;
    std::string device;
    std::vector<int> hw;
    int img_h = 112;
    int img_w = 112;

    int res;
    while((res = getopt(argc, argv, "d:t:g:r:h")) != -1)
    {
        switch(res)
        {
            case 'd':
                device = optarg;
                break;
            case 't':
                tm_file = optarg;
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
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "    [-t tm_file] [-i image_file]\n"
                          << "    [-g img_h,img_w] [-r repeat_count]\n";
                return 0;
            default:
                break;
        }
    }

    // load model
    if(tm_file.empty())
    {
        tm_file = root_path + DEF_MODEL;
        std::cout << "model file not specified,using " << tm_file << " by default\n";
    }

    image_file = root_path + DEF_IMAGE;
    image_file2 = root_path + DEF_IMAGE2;
    image_file3 = root_path + DEF_IMAGE3;
    std::cout << "image1 file not specified,using " << image_file << " by default\n";
    std::cout << "image2 file not specified,using " << image_file2 << " by default\n";
    std::cout << "image3 file not specified,using " << image_file3 << " by default\n";

    // check file
    if((!check_file_exist(tm_file) or !check_file_exist(image_file) or !check_file_exist(image_file2) or !check_file_exist(image_file3)))
    {
        return -1;
    }

    float *featureA;
    float *featureB;
    float *featureC;

    featureA = (float *)malloc(sizeof(float)*FEATURESIZE);
    featureB = (float *)malloc(sizeof(float)*FEATURESIZE);
    featureC = (float *)malloc(sizeof(float)*FEATURESIZE);

    // init
    init_tengine();
    std::cout << "tengine library version: " << get_tengine_version() << "\n";
    if(request_tengine_version("1.0") < 0)
        return false;

    // start to run
    if(!run_tengine_library(tm_file.c_str(), image_file, img_h, img_w, repeat_count, device, featureA))
        return -1;

    printf("run second image\n");
    // start to run
    if(!run_tengine_library(tm_file.c_str(), image_file2, img_h, img_w, repeat_count, device, featureB))
        return -1;

    // start to run
    if(!run_tengine_library(tm_file.c_str(), image_file3, img_h, img_w, repeat_count, device, featureC))
        return -1;

    /* calculte similarity of two face features */
    float similarity_sameid = cosine_dist(featureA, featureB, FEATURESIZE);
    float similarity_diffid = cosine_dist(featureA, featureC, FEATURESIZE);

    std::cout << "similarity_sameid > 60% : " << similarity_sameid << std::endl;
    std::cout << "similarity_diffid < 10% : " << similarity_diffid << std::endl;

    const float sameid_value = 0.60;
    const float diffid_value = 0.10;
    if ((similarity_sameid > sameid_value) && (similarity_diffid < diffid_value)) {
        std::cout << "test pass" << std::endl;
    }
    else
    {
        std::cout << "test fail" << std::endl;
	return -1;
    }

    free(featureA);
    free(featureB);
    free(featureC);
    release_tengine();
    std::cout << "ALL TEST DONE\n";
    return 0;
}
