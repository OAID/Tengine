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

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include "tengine_c_api.h"
#include "tengine_config.hpp"

// densenet.pb download form:
// https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz

// test_data in: tengine-Server:/home/public/tf_models/densenet

const char* model_file = "../densenet/densenet.pb";
const char* input_file = "../densenet/densenet_inp";
const char* output_file = "../densenet/densenet_out";

int img_h = 224;
int img_w = 224;

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

void get_data(void* buffer, int datasize, const char* fname)
{
    // read data
    FILE* data_fp = fopen(fname, "rb");
    if(!data_fp)
        printf("data can not be open\n");

    size_t n = fread(buffer, sizeof(float), datasize, data_fp);
    if(( int )n < datasize)
        printf("data read error\n");

    fclose(data_fp);
}

void maxerr(float* pred, float* gt, int size)
{
    float maxError = 0.f;
    for(int i = 0; i < size; i++)
    {
        maxError = MAX(( float )fabs(gt[i] - *(pred + i)), maxError);
    }
    printf("====================================\n");
    printf("maxError is %f\n", maxError);
    printf("====================================\n");
}
using namespace TEngine;

int main(int argc, char* argv[])
{
    /* prepare input data */
    int img_size = img_h * img_w * 3;

    float* input_data = ( float* )malloc(img_size * sizeof(float));
    get_data(input_data, img_size, input_file);

    float* output_gt = ( float* )malloc(1001 * sizeof(float));
    get_data(output_gt, 1001, output_file);

    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "tensorflow", model_file);
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

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */
    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }

    /* run the graph */
    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");
    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    // warm up
    run_graph(graph, 1);

    struct timeval t0, t1;
    float avg_time = 0;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        run_graph(graph, 1);
        gettimeofday(&t1, NULL);
        float each_time = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += each_time;
        if(i < 10)
            std::cout << "repeat_idx:" << i << "\t" << each_time << "ms\n";
    }

    tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    float* data = ( float* )(get_tensor_buffer(output_tensor));
    maxerr(data, output_gt, 1001);

    std::cout << "avg_time = \t" << avg_time / repeat_count << "ms\n\n";
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);
    free(output_gt);

    postrun_graph(graph);
    destroy_graph(graph);

    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
