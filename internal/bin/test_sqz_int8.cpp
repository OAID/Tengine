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
 * Copyright (c) 2019, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "tengine_operations.h"
#include "tengine_c_api.h"
#include "tengine_config.hpp"
#include <sys/time.h>
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

int repeat_count = 100;


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

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    image img = imread(image_file);

    image resImg = resize_image(img, img_w, img_h);
    resImg = rgb2bgr_premute(resImg);
    float* img_data = ( float* )resImg.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
            {   
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }   
}

int main(int argc, char* argv[])
{
    int img_h = 227;
    int img_w = 227;
    int res;
    std::string model_file = "./models/sqz_int8.tmfile";
    while((res = getopt(argc, argv, "m:r:")) != -1) 
    {   
        switch(res)
        {   
            case 'm':
                model_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }   
    }   


    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tengine", model_file.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    /* set input shape */
    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
 
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d \n", node_idx, tensor_idx);
        return -1;
    }
    int dims[] = {1, 3,img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    /* setup input buffer */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);
    int img_size = img_h * img_w * 3;
    float in_scale = 0;
    int in_zero = 0;
    get_tensor_quant_param(input_tensor,&in_scale,&in_zero,1);
    printf("intput scale is %f,input zero point is %d\n",in_scale,in_zero);
    //quant the input data
    int8_t * input_s8 = (int8_t*)malloc(sizeof(int8_t) * img_h * img_w * 3);
    for(int i = 0; i < img_size; ++i)
    {
        input_s8[i] = round(input_data[i] / in_scale);
    }
    //set the input data type 
    set_tensor_data_type(input_tensor,TENGINE_DT_INT8);
    if(set_tensor_buffer(input_tensor, input_s8, 3 * img_h * img_w) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }
    if(prerun_graph(graph) < 0){
        printf("Prerun Error\n"); 
        return -1 ;
    }
    
    run_graph(graph, 1);

    run_graph(graph,1);
    run_graph(graph,1);
    run_graph(graph,1);
    run_graph(graph,1);

    struct timeval t0, t1;
    float avg_time = 0.f;
    float min_time = 0x7fffffff;
    float max_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        //get_input_data(image_file, input_data, img_h, img_w, mean, scale);
        //set_tensor_buffer(input_tensor, input_data, img_size * 4);

        gettimeofday(&t0, NULL);
        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "Run graph failed\n";
            return false;
        }
        gettimeofday(&t1, NULL);

        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;
        if(mytime > max_time)
            max_time = mytime;
        else
            max_time = max_time;
        if(mytime < min_time)
            min_time = mytime;
        else
            min_time = min_time;
    }
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";
    std::cout << "--------------------------------------\n";


    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }

    int8_t* out_data_s8 = (int8_t* )(get_tensor_buffer(output_tensor));
    int count = get_tensor_buffer_size(output_tensor);
    float * out_data_fp32 = (float*) malloc(count * sizeof(float));
    float out_scale = 1.f;
    int out_zero = 0;
    
    get_tensor_quant_param(output_tensor,&out_scale,&out_zero,1);
    printf("output scale is %f\n",out_scale);


    //dequant the output data
    for(int i = 0; i < count ; i ++)
    {
        out_data_fp32[i] = out_data_s8[i] * out_scale;
    }
    float* end = out_data_fp32 + count;

    std::vector<float> result(out_data_fp32, end);
    std::vector<int> top_N = Argmax(result, 5);
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);
    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx] << "\"\n";
    }
      
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);
    free(input_s8);
    free(out_data_fp32);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
