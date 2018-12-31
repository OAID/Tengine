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
#include <time.h>

#include "tengine_c_api.h"
#include "cpu_device.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "common_util.hpp"

const char* text_file = "./models/sqz.prototxt";
const char* model_file = "./models/squeezenet_v1.1.caffemodel";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

using namespace TEngine;

int repeat_count = 10;
void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void PrintTopLabels(const char* label_file, float* data)
{
    // load labels
    std::vector<std::string> labels;
    LoadLabelFile(labels, label_file);

    float* end = data + 1000;
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
#if 1
    cv::Mat img = cv::imread(image_file, -1);

    if(img.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
        for(int w = 0; w < img_w; w++)
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
#endif
}

void dump_perf_stat(struct perf_info** info_array, int number)
{
    printf("\n================================\n");

    for(int i = 0; i < number; i++)
    {
        struct perf_info* info = info_array[i];

        float avg_time = info->total_time * 1.0 / info->count;
        printf("node: %-40s\tcount: %d  time:(%.2f/%u/%u)\t\tdev:%s\n", info->name, info->count, avg_time, info->max,
               info->min, info->dev_name);
    }

    printf("\n================================\n");
}

int main(int argc, char* argv[])
{
    int res;

    while((res = getopt(argc, argv, "d:r:")) != -1)
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

    int img_h = 227;
    int img_w = 227;

    std::string model_name = "squeeze_net";

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);

    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);

    init_tengine();

    std::cout << "run-time library version: " << get_tengine_version() << "\n";

    if(request_tengine_version("0.9") < 0)
        return -1;

    const struct cpu_info* p_info = get_predefined_cpu("rk3399");
    int a72_list[] = {4, 5};

    set_online_cpu(( struct cpu_info* )p_info, a72_list, sizeof(a72_list) / sizeof(int));
    create_cpu_device("a72", p_info);

    const struct cpu_info* p_info1 = get_predefined_cpu("rk3399");
    int a53_list[] = {0, 1, 2, 3};

    set_online_cpu(( struct cpu_info* )p_info1, a53_list, sizeof(a53_list) / sizeof(int));
    create_cpu_device("a53", p_info1);

    graph_t graph = create_graph(nullptr, "caffe", text_file, model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
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

    node_t pool_node = get_graph_node(graph, "pool1");

    if(set_node_device(pool_node, "a53") < 0)
    {
        std::cerr << "Set node device: " << get_tengine_errno() << "\n";
    }

    release_graph_node(pool_node);

    /* run the graph */

    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    // warm up
    run_graph(graph, 1);

    if(do_graph_perf_stat(graph, GRAPH_PERF_STAT_ENABLE) < 0)
    {
        std::cerr << "PERF STAT ENABLE: " << get_tengine_errno() << "\n";
    }

    if(do_graph_perf_stat(graph, GRAPH_PERF_STAT_START) < 0)
    {
        std::cerr << "PERF STAT START: " << get_tengine_errno() << "\n";
    }

    printf("REPEAT COUNT= %d\n", repeat_count);
    unsigned long start_time = get_cur_time();

    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);

    unsigned long end_time = get_cur_time();

    unsigned long off_time = end_time - start_time;

    std::printf("Repeat [%d] time %.2f us per RUN. used %lu us\n", repeat_count, 1.0f * off_time / repeat_count,
                off_time);

    if(do_graph_perf_stat(graph, GRAPH_PERF_STAT_STOP) < 0)
    {
        std::cerr << "PERF STAT STOP: " << get_tengine_errno() << "\n";
    }

    struct perf_info** info;
    int info_size = 100;

    info = ( struct perf_info** )malloc(info_size * sizeof(void*));

    int ret_number = get_graph_perf_stat(graph, info, info_size);

    std::cout << "total received " << ret_number << " records\n";

    if(ret_number > 0)
    {
        dump_perf_stat(info, ret_number);
    }

    free(info);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* data = ( float* )get_tensor_buffer(output_tensor);
    PrintTopLabels(label_file, data);
    std::cout << "--------------------------------------\n";

    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);

    if(do_graph_perf_stat(graph, GRAPH_PERF_STAT_DISABLE) < 0)
    {
        std::cerr << "PERF STAT DISABLE: " << get_tengine_errno() << "\n";
    }

    postrun_graph(graph);
    destroy_graph(graph);

    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
