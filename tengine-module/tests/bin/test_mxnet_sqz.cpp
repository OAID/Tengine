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

const char* text_file = "./models/squeezenet_v1.1-symbol.json";
const char* model_file = "./models/squeezenet_v1.1-0000.params";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

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

    // const char * model_name="squeezenet";
    int img_h = 224;
    int img_w = 224;

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);

    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);

    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return -1;

#if 0
   if(load_model(model_name,"mxnet",text_file,model_file)<0)
       return 1; 

   std::cout<<"Load model successfully\n";
#endif
    graph_t graph = create_graph(nullptr, "mxnet", text_file, model_file);

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

    int dims[] = {1, 3, img_h, img_w};

    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
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

    for(int i = 0; i < 10; i++)
        run_graph(graph, 1);

    // benchmark start here

    printf("REPEAT COUNT= %d\n", repeat_count);

    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;
    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
                off_time);

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

    int count = get_tensor_buffer_size(output_tensor) / 4;

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    float* end = data + count;

    std::vector<float> result(data, end);

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

    postrun_graph(graph);

    destroy_graph(graph);

    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
