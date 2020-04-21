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

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"
#include "common_util.hpp"
#include "image_process.hpp"

//const char* text_file = "../models/1.json";
const char* text_file = "../models/mobilenet_int8.tmfile";
const char* image_file = "./images/cat.jpg";
const char* label_file = "../models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

using namespace TEngine;

int repeat_count = 1;

void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

int main(int argc, char* argv[])
{
    int res;

    while((res = getopt(argc, argv, "r:")) != -1)
    {
        switch(res)
        {
            case 'r':
                repeat_count = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    // const char * model_name="mobilenet";
    int img_h = 224;
    int img_w = 224;

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 0.017);
    int img_size = img_h * img_w * 3;
    float in_scale = 0;
    int in_zero = 0;
    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tengine", text_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* get input tensor */
    int node_idx = 0;
    int tensor_idx = 0;

    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }
    
	get_tensor_quant_param(input_tensor,&in_scale,&in_zero,1);
	
	int8_t * input_s8 = (int8_t*)malloc(sizeof(int8_t) * img_size);
	
	for(int i = 0; i < img_size;i++)
	{
		input_s8[i] = round(input_data[i] / in_scale);
	} 

    int dims[] = {1, 3, img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);
	set_tensor_data_type(input_tensor,TENGINE_DT_INT8);
    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_s8, 3 * img_h * img_w) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    /* run the graph */
    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    // benchmark start here

    //printf("REPEAT COUNT= %d\n", repeat_count);

    //unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    //unsigned long end_time = get_cur_time();

    //unsigned long off_time = end_time - start_time;
    //std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
    //            off_time);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);

    if(output_tensor == nullptr)
    {
        std::printf("Cannot find output tensor , node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dim_size = get_tensor_shape(output_tensor, dims, 4);

    if(dim_size < 0)
    {
        printf("Get output tensor shape failed\n");
        return -1;
    }

    printf("output tensor shape: [");

    for(int i = 0; i < dim_size; i++)
        printf("%d ", dims[i]);

    printf("]\n");

    int count = get_tensor_buffer_size(output_tensor);
	int8_t * out_data_s8 = (int8_t*)(get_tensor_buffer(output_tensor));
	float *out_data_fp32 = (float*) malloc(count * sizeof(float));
	float out_scale = 1.f;
	int out_zero = 0;
	get_tensor_quant_param(output_tensor,&out_scale,&out_zero,1);

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

    //
    std::vector<std::string> cla_res(5);
    cla_res[0] = "n02123159 tiger cat";
    cla_res[1] = "n02119789 kit fox, Vulpes macrotis";
    cla_res[2] = "n02119022 red fox, Vulpes vulpes";
    cla_res[3] = "n02113023 Pembroke, Pembroke Welsh corgi";
    cla_res[4] = "n02123045 tabby, tabby cat";
    int miss_cnt = 0;
    //
    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx] << "\"\n";
        
        if(labels[idx]!=cla_res[i]) miss_cnt++;
    }
    //
    if(miss_cnt==0) printf("pass\n");
    else printf("fail\n");
    //

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    postrun_graph(graph);

    destroy_graph(graph);

    free(input_data);
	free(input_s8);
    free(out_data_fp32);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
