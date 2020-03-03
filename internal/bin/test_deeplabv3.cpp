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

// nasnet_mobile.pb download form:
// https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz
// model in: tengine-Server:/home/public/tf_models/nasnet_mobile

const char* model_file = "./models/deeplabv3_cutoff_2.pb";
const char* label_file = "./models/synset_words.txt";
const char* image_file = "./tests/images/bike.jpg";
const char* filename = "./models/input.txt";

int img_h = 513;
int img_w = 513;
const float channel_mean[3] = {128, 128, 128};
float scale = 0.0078431;
using namespace TEngine;
void inline dump_kernel_value(const tensor_t tensor, const char* dump_file_name)
{
    std::ofstream of(dump_file_name, std::ios::out);
    int kernel_dim[4];
    int dim_len = 0;
    dim_len = get_tensor_shape(tensor, kernel_dim, 4);
    int data_couts = 1;
    for(int ii = 0; ii < dim_len; ++ii)
    {
        data_couts *= kernel_dim[ii];
    }

    const float* tmp_data = ( const float* )get_tensor_buffer(tensor);
    char tmpBuf[1024];
    int iPos = 0;
    for(int ii = 0; ii < data_couts; ++ii)
    {
        iPos += sprintf(&tmpBuf[iPos], "%.18e", tmp_data[ii]);
        of << tmpBuf << std::endl;
        iPos = 0;
    }

    of.close();
}
void LoadLabelFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream labels(fname);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void getdatanptxt(int buf_size,float* w_buf,const char* filename)
{
    char buffer[256];
    std::fstream outfile;
    outfile.open(filename,std::ios::in);
    unsigned int a=0;
    while(a<(buf_size/sizeof(float)))
    {
        outfile.getline(buffer,256);
        w_buf[a]=atof(buffer);
        a++;
    }
    outfile.close();
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, float scale)
{
    /*
    image im = imread(image_file);
    
    float resize_ratio = (513.0 / std::max(im.h, im.w));
    img_w = (int)(resize_ratio * im.w);
    img_h = (int)(resize_ratio * im.h);
    printf("Image size: %d %d \n", img_w, img_h);
    image resImg = resize_image(im, img_w, img_h);

    image mask = make_image(513, 513, 3);

    for(int c = 0; c < 3; c++){
        for(int i = 0; i < 513; i++){
            for(int j = 0; j < 513; j++){
                if(i < img_w && j < img_h)
                    mask.data[c*513*513 + i*513 + j] = resImg.data[c*img_w*img_h + i*img_w + j];
                else{
                    mask.data[c*513*513 + i*513 + j] = 128;
                }
            }
        }
    }

    int index = 0;
    img_w = 513;
    img_h = 513;
    for(int h = 0; h < 513; h++)
    {
        for(int w = 0; w < 513; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[index] = scale * ( mask.data[c * img_h * img_w + h * img_w + w] ) - 1;
                index++;
            }
        }
    }
    free_image(resImg);
    free_image(mask);
    */
    int buf_size = 513*513*3*sizeof(float);
    getdatanptxt(buf_size, input_data, filename);
    for(int i = 0; i < buf_size; i++){
        //printf("%f ",input_data[i]);
        //if(input_data[i] >= 1 || input_data[i] <= -1){
            //input_data[i] -= 0.1;

        //    printf("%f \n",input_data[i]);
        //}
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

int main(int argc, char* argv[])
{
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
    int dims[] = {1, img_h, img_w, 3};
    set_tensor_shape(input_tensor, dims, 4);

    /* setup input buffer */
    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, channel_mean, scale);

    if(set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
    }
    //dump_kernel_value(input_tensor, "Input_tensor.txt");    
    /* run the graph */
    //dump_graph(graph);
    if(prerun_graph(graph) < 0){
        printf("Prerun Error\n"); 
        return 0 ;
    }
    //dump_graph(graph);
    //printf("Prerun success\n");
    run_graph(graph, 1);

    // tensor_t output_tensor = get_graph_output_tensor(graph, node_idx, tensor_idx);
    std::string output_tensor_name = "logits/semantic/Conv2D";
    tensor_t output_tensor = get_graph_tensor(graph, output_tensor_name.c_str());
    if(NULL == output_tensor)
    {
        std::cout << "output tensor is NULL\n";
    }
    int dim_size = get_tensor_shape(output_tensor, dims, 4);
    if(dim_size < 0)
    {
        printf("get output tensor shape failed\n");
        return -1;
    }
    std::cout << "dim size is :" << dim_size << "\n";
    printf("output tensor shape: [");
    int total = 1;
    for(int i = 0; i < dim_size; i++){
        printf("%d ", dims[i]);
        total *= dims[i];
    }
    printf("]\n");



    //int count = get_tensor_buffer_size(output_tensor) / 4;

    float* data = ( float* )(get_tensor_buffer(output_tensor));
    //float* end = data + count;
    
    for(int i = 0; i < 20; i++){
        printf("%f \n", data[i]);
    }
    
    /*
    std::vector<float> result(data, end);
    std::vector<int> top_N = Argmax(result, 5);
    std::vector<std::string> labels;

    LoadLabelFile(labels, label_file);

    for(unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx - 1] << "\"\n";
    }
    */
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
