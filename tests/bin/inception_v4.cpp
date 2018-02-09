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
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "prof_utils.hpp"
#include "tengine_c_api.h"
#include "image_process.hpp"
#include "common_util.hpp"
#include "tengine_config.hpp"
#include "prof_utils.hpp"
#include "prof_record.hpp"

using namespace TEngine;
const char *label_file = "./tests/data/synset_words.txt";

void LoadLabelFile(std::vector<std::string> &result, const char *fname)
{
    std::ifstream labels(fname);

    std::string line;
    while (std::getline(labels, line))
        result.push_back(line);
}

void get_input_data_inceptionV4(std::string& image_file, float* input_data, int img_h,  int img_w)
{
    cv::Mat img = cv::imread(image_file, -1);

    if (img.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - 127.5)/127.5;
                img_data++;
            }
        }
    }
}
int main(int argc, char *argv[])
{
    if(argc!=3)
    {
        std::cout<<"[Usage]: "<<argv[0]<<" <Tengine_model_dir> <test.jpg>\n";
        return 0;
    }
    std::string model_dir=argv[1];
    std::string image_file=argv[2];
    // std::string model_dir="../Tengine_models/";
    // std::string image_file="./examples/resnet50/images/bike.jpg";
    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;

    // load model
    const char* model_name="inception_v4";
    std::string proto_name_ =model_dir+"/inception_v4/inception_v4.prototxt";
    std::string mdl_name_ = model_dir+"/inception_v4/inception_v4.caffemodel";
    if (load_model(model_name, "caffe", proto_name_.c_str(), mdl_name_.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";

    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    // input
    int img_h = 299;
    int img_w = 299;
    int img_size = img_h * img_w * 3;
    float *input_data = (float *)malloc(sizeof(float) * img_size);
    get_input_data_inceptionV4(image_file, input_data, img_h,  img_w);
  
  /* set input and output node*/
    const char *input_node_name = "input";
    set_graph_input_node(graph, &input_node_name, 1);

    const char *input_tensor_name = "data";
    tensor_t input_tensor = get_graph_tensor(graph, input_tensor_name);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4);

    /* run the graph */
    prerun_graph(graph);
    run_graph(graph,1); 
    //dump_graph(graph);
    free(input_data);

    
    //    printf("REPEAT COUNT= %d\n",repeat_count);

    //    unsigned long start_time=get_cur_time();

    //    for(int i=0;i<repeat_count;i++)
    //        run_graph(graph,1);

    //    unsigned long end_time=get_cur_time();

    //    printf("Repeat [%d] times %.2f per RUN, used [%lu] us\n",repeat_count,1.0f*(end_time-start_time)/repeat_count,
    //                     end_time-start_time);


    tensor_t output_tensor = get_graph_tensor(graph, "prob");
    float *data = (float *)get_tensor_buffer(output_tensor);
    float *end = data + 1000;

    std::vector<float> result(data, end);

    std::vector<int> top_N = Argmax(result, 5);

    std::vector<std::string> labels;

    LoadLabelFile(labels, label_file);

    for (unsigned int i = 0; i < top_N.size(); i++)
    {
        int idx = top_N[i];

        std::cout << std::fixed << std::setprecision(4) << result[idx] << " - \"";
        std::cout << labels[idx] << "\"\n";
     }
    postrun_graph(graph);

    ProfRecord * prof=ProfRecordManager::Get("simple");

    if(prof)
      prof->Dump(1);

 

   destroy_runtime_graph(graph);
   remove_model(model_name);


   std::cout<<"ALL TEST DONE\n";


    return 0;
}


   

