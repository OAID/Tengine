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
#include "tengine_operations.h"

const char* model_proto = "./models/shufflenet_v2.prototxt";
const char* model_param = "./models/shufflenet_v2.caffemodel";
const char* image_file = "./tests/images/cat.jpg";
const float channel_mean[3] = {0, 0, 0};

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
void getdatacompare(const char* filename,const char* filename1)
{
    char buffer[256];
    char buffer1[256];
    std::fstream outfile;
    std::fstream outfile1;
    std::vector<float> f_vec={};
    std::vector<float> f_vec1={};
    outfile.open(filename,std::ios::in);

    while(outfile.getline(buffer,256))
    {
        f_vec.push_back(atof(buffer));
    }
    outfile1.open(filename1,std::ios::in);
    while(outfile1.getline(buffer1,256))
    {
        f_vec1.push_back(atof(buffer1));
    }
    float losssum=0;
    for(unsigned int i=0;i<f_vec.size();i++)
    {
        losssum+=fabs((f_vec[i]-f_vec1[i]));
    }
    float avg_loos=losssum/f_vec.size();
    if(avg_loos<0.0002)
    {
        std::cout<<"!!!!COMPARE PASS!!!!\n";
    }else
    {
        std::cout<<"!!!!COMPARE NOPASS!!!!\n";
    }
    outfile.close();
    outfile1.close();
}

int main(int argc, char* argv[])
{
    int img_h = 224;
    int img_w = 224;

    float* input_data = ( float* )malloc(sizeof(float) * img_h * img_w * 3);
    get_input_data(image_file, input_data, img_h, img_w, channel_mean, 1);
    init_tengine();

    graph_t graph = create_graph(nullptr, "caffe",model_proto,model_param);
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return 1;
    }

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dim[4] = {1, 3, 224, 224};

    set_tensor_shape(input_tensor, dim, 4);
    int input_size = get_tensor_buffer_size(input_tensor);

    set_tensor_buffer(input_tensor, input_data, input_size);

    prerun_graph(graph);

    // dump_graph(graph);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    dump_kernel_value(output_tensor, "./tests/data/test_shuffle_out.txt");
    getdatacompare("./tests/data/test_shuffle_out.txt","caffe_shufflenet_out.txt");
    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
