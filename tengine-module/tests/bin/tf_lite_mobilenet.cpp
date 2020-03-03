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

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>

#include "tengine_operations.h"
#include "tengine_c_api.h"

//#define DEF_MODEL "models/mobilenet_v1_1.0_224.tflite"
#define DEF_MODEL "models/mobilenet_v2_1.0_224.tflite"
#define DEF_LABEL "models/imagenet_slim_labels.txt"
#define DEF_IMAGE "tests/images/cat.jpg"

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w)
{
    image im = imread(image_file);

    image resImg = resize_image(im, img_w, img_h);

    int index = 0;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[index] = (resImg.data[c * img_h * img_w + h * img_w + w] - 128) / 255;
                index++;
            }
        }
    }
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

void LoadLabelFile(std::vector<std::string>& result)
{
    std::ifstream labels(DEF_LABEL);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void post_process(tensor_t tensor)
{
    float* predictions = ( float* )get_tensor_buffer(tensor);

    std::vector<std::string> labels;
    LoadLabelFile(labels);

    std::vector<float> result;
    for(int i = 0; i < 1001; i++)
        result.push_back(predictions[i]);

    std::vector<int> top_N = Argmax(result, 5);

    for(size_t i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];
        std::cout << std::fixed << std::setprecision(5) << result[idx] << " - \"" << labels[idx] << "\"\n";
    }
}

int main(int argc, char* argv[])
{
    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(0, "tflite", DEF_MODEL);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!\n";
        return 1;
    }
    std::cout << "create graph done!\n";

    // dump_graph(graph);

    // input
    int img_h = 224;
    int img_w = 224;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(!check_tensor_valid(input_tensor))
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return 1;
    }

    int dims[] = {1, img_h, img_w, 3};
    set_tensor_shape(input_tensor, dims, 4);
    if(prerun_graph(graph) != 0)
    {
        std::cout << "prerun _graph failed\n";
        return -1;
    }

    int repeat_count = 1;
    if(argc > 1)
        repeat_count = atoi(argv[1]);
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    // warm up
    get_input_data(DEF_IMAGE, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size);
    if(run_graph(graph, 1) != 0)
    {
        std::cout << "run _graph failed\n";
        return -1;
    }

    struct timeval t0, t1;
    float total_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
    }
    std::cout << "--------------------------------------\n";
    std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";

    tensor_t predictions = get_graph_output_tensor(graph, 0, 0);

    post_process(predictions);

    postrun_graph(graph);
    free(input_data);

    destroy_graph(graph);

    return 0;
}
