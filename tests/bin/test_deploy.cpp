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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <iomanip>
#include <string>
#include "tengine_c_api.h"
#include <sys/time.h>

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        std::cout << "[Usage]: " << argv[0] << " <proto> <caffemodel> <size> \n";
        return 0;
    }

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return -1;

    std::string proto_name_ = argv[1];
    std::string mdl_name_ = argv[2];
#if 0
    if (load_model(model_name, "caffe", proto_name_.c_str(), mdl_name_.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";
#endif
    // create graph
    graph_t graph = create_graph(NULL, "caffe", proto_name_.c_str(), mdl_name_.c_str());
    if(graph == nullptr)
    {
        std::cout << "create graph0 failed\n";
        return -1;
    }

    // input
    int img_h = atoi(argv[3]);
    int img_w = img_h;
    if(argc == 5)
        img_w = atoi(argv[4]);
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    for(int i = 0; i < img_size; i++)
        input_data[i] = (i % 123) * 0.017;

    const char* input_tensor_name = "data";
    tensor_t input_tensor = get_graph_tensor(graph, input_tensor_name);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    // if use gpu
    int use_gpu = 0;
    const char* gpu_flag = std::getenv("USE_GPU");
    if(gpu_flag)
        use_gpu = atoi(gpu_flag);
    if(use_gpu)
        set_graph_device(graph, "acl_opencl");
    //

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

    struct timeval t0, t1;
    float avg_time = 0.f;
    set_tensor_buffer(input_tensor, input_data, img_size * 4);

    gettimeofday(&t0, NULL);

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    avg_time += mytime;

    std::cout << "--------------------------------------\n";
    std::cout << "repeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";
    free(input_data);
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);

    std::cout << "ALL TEST DONE\n";
    release_tengine();
    return 0;
}
