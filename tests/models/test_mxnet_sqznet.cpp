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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define VXDEVICE  "VX"

const char* model_file = "./models/xception_sim.tmfile";
const char* image_file = "./catcat.bin";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

int repeat_count = 1;

static double get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
}

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

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

bool get_input_bin_data(const char* image_file, void* input_data, int input_length)
{
    FILE* fp = fopen(image_file, "rb");
    
    if(fp == nullptr)
    {
        std::cout << "Open input data file failed: " << image_file << "\n";
        return false;
    }

    int res = fread(input_data, 1, input_length, fp);
    if(res != input_length)
    {
        std::cout << "Read input data file failed: " << image_file << "\n";
        return false;
    }
    fclose(fp);
    return true;
}

int main(int argc, char* argv[])
{
    int res;

    while((res = getopt(argc, argv, "i:r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            case 'i':
                image_file = optarg ;
            default:
                break;
        }
    }

    int img_h = 299;
    int img_w = 299;

    /* prepare input data */
    // struct stat statbuf;
    // stat(image_file, &statbuf);
    int input_length = 3* img_h * img_w * 4;

    // if(input_length != 3* img_h * img_w * 4)
    // {
    //     printf("Input bin file size : %d is not correct\n",input_length );
    //     return -1 ;
    // }

    // void* inputbin_data = malloc(input_length);

    // if(!get_input_bin_data(image_file, inputbin_data, input_length)){
    //     printf(" Get input bin file data failed!\n");
    //     return -1;
    // }

    float* udata = (float *) malloc(input_length);
    // float *p = (float *)inputbin_data ;
    int i = 0 ;
    for(i = 0 ; i < input_length/4 ; i++)
    {
        udata[i] = 1 ;
    }

    /* prepare input data */
    init_tengine();

    std::cout << "model name : " << model_file << "\n";
    
    std::cout << "run-time library version: " << get_tengine_version() << "\n";

    if(request_tengine_version("1.0") < 0)
        return -1;

    // if(load_tengine_plugin(VXDEVICE,"libvxplugin.so","vx_plugin_init")<0)
    // {
    //     printf("load tee plugin failed\n");
    //     return 1;
    // }

    graph_t graph = create_graph(nullptr, "tengine", model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    // set_graph_device(graph , VXDEVICE);   

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;

    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);

    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, 3,img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);

    // dump_graph(graph);

    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, udata, 3 * img_h * img_w*4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    std::cout << "PRERUN NOW ....\n";

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

    double total_time, min_time, max_time;
    min_time = 999999999;
    max_time = 0;
    total_time = 0;

    for(int i = 0; i < repeat_count; i++)
    {
        double start_time = get_cur_time();
        run_graph(graph, 1);
        double end_time = get_cur_time();
        double cur_time = end_time - start_time;

        total_time += cur_time;
        if (cur_time > max_time)
            max_time = cur_time;
        if (cur_time < min_time)
            min_time = cur_time;

        printf("Cost %.3f ms\n", cur_time);
    }
    printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time, total_time / repeat_count);    

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);

    if(output_tensor == nullptr)
    {
        std::printf("Cannot find output tensor , node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int count = get_tensor_buffer_size(output_tensor);

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    for(int i=0;i<10;i++)
    {
        printf("outdata: %f\n",data[i]);
    }
    float* end = data + count;
    std::vector<float> result(data, end);

    std::vector<int> top_N = Argmax(result, 5);

    std::cout << "\nIndex" << " - "<< "Score:\n";
    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << idx <<" - " << std::fixed << std::setprecision(4) << (result[idx]*0.160932) << "\n";
    }
    
    
    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    // free(inputbin_data);

    std::cout << "\nALL TEST DONE\n";

    release_tengine();
    return 0;
}
