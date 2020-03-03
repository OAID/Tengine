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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>

#include "tengine_c_api.h"
#include "tengine_operations.h"
#include "common_util.hpp"

#define TEST_CAFFE
//#define TEST_TENGINE

const char* text_file = "./models/sqz.prototxt";
const char* model_file = "./models/squeezenet_v1.1.caffemodel";
const char* image_file = "./tests/images/cat.jpg";
const char* label_file = "./models/synset_words.txt";

const float channel_mean[3] = {104.007, 116.669, 122.679};

using namespace TEngine;

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

bool get_file_mem_size(const char* fname, void** addr, int* size)
{
    int fd = open(fname, O_RDONLY);

    if(fd < 0)
        return false;

    struct stat stat_buf;

    fstat(fd, &stat_buf);

    void* mem = mmap(NULL, stat_buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if(mem == MAP_FAILED)
        return false;

    *addr = mem;
    *size = stat_buf.st_size;

    return true;
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
#if 1
    image img = imread(image_file);

    if(img.data == 0)
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }
    image resize_img = resize_image(img, img_h, img_w);
    float* img_data = ( float* )resize_img.data;
    int hw = img_h * img_w;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale;
                img_data++;
            }
#endif
}

int main(int argc, char* argv[])
{
    std::string device;

    int img_h = 227;
    int img_w = 227;

    std::string model_name = "squeeze_net";

    /* prepare input data */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);

    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);

    init_tengine();

    std::cout << "run-time library version: " << get_tengine_version() << "\n";

    if(request_tengine_version("1.3.2") < 0)
        return -1;

#ifdef TEST_CAFFE
    int text_size, bin_size;
    void *text_mem, *bin_mem;

    if(!get_file_mem_size(text_file, &text_mem, &text_size))
    {
        std::cerr << "cannot load " << text_file << "\n";
        return -1;
    }

    if(!get_file_mem_size(model_file, &bin_mem, &bin_size))
    {
        std::cerr << "cannot load " << model_file << "\n";
        return -1;
    }

    graph_t graph = create_graph(nullptr, "caffe:m", ( const char* )text_mem, text_size, bin_mem, bin_size);
#endif

#ifdef TEST_TENGINE
    int bin_size;
    void* bin_mem;
    model_file = "./models/sqz.tm";

    if(!get_file_mem_size(model_file, &bin_mem, &bin_size))
    {
        std::cerr << "cannot load " << model_file << "\n";
        return -1;
    }

    graph_t graph = create_graph(nullptr, "tengine:m", ( const char* )bin_mem, bin_size);
#endif

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

    /* run the graph */

    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* data = ( float* )get_tensor_buffer(output_tensor);
    PrintTopLabels(label_file, data);
    std::cout << "--------------------------------------\n";

    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);

    postrun_graph(graph);
    destroy_graph(graph);

    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
