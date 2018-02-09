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

const char *text_file = "./tests/data/resnet50/resnet50.prototxt";
const char *model_file = "./tests/data/resnet50/resnet50.caffemodel";
const char *label_file = "./tests/data/synset_words.txt";

using namespace TEngine;

int repeat_count = 1;

void LoadLabelFile(std::vector<std::string> &result, const char *fname)
{
    std::ifstream labels(fname);

    std::string line;
    while (std::getline(labels, line))
        result.push_back(line);
}

void get_input_data(const char *image_file, float *data, int img_h, int img_w)
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
                data[c * hw + h * img_w + w] = *img_data - 127.5;
                img_data++;
            }
        }
    }
}

int main(int argc, char *argv[])
{

    const char *image_file = "./examples/resnet50/images/bike.jpg";
    int res;

    while ((res = getopt(argc, argv, "er:")) != -1)
    {
        switch (res)
        {
        case 'e':
            TEngineConfig::Set("exec.engine", "event");
            break;
        case 'r':
            repeat_count = strtoul(optarg, NULL, 10);
            break;
        default:
            break;
        }
    }
    const char *model_name = "resnet50";
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;
    if (load_model(model_name, "caffe", text_file, model_file) < 0)
        return 1;
    std::cout << "Load model successfully\n";
    //dump_model(model_name);
    graph_t graph = create_runtime_graph("graph0", model_name, NULL);
 

    int img_h = 224;
    int img_w = 224;

    /* prepare input data */
    int in_size=img_w*img_h*3;
    float* input_data=(float*)malloc(sizeof(float)*in_size);
    get_input_data(image_file, input_data, img_h,  img_w);


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
    

  run_graph(graph,1); //warm up

   printf("REPEAT COUNT= %d\n",repeat_count);

   unsigned long start_time=get_cur_time();

   for(int i=0;i<repeat_count;i++)
       run_graph(graph,1);

   unsigned long end_time=get_cur_time();

   printf("Repeat [%d] times %.2f per RUN, used [%lu] us\n",repeat_count,1.0f*(end_time-start_time)/repeat_count,
                    end_time-start_time);

    free(input_data);

    tensor_t mytensor1 = get_graph_tensor(graph, "prob");
    float *data1 = (float *)get_tensor_buffer(mytensor1);
    float *end = data1 + 1000;

    std::vector<float> result(data1, end);

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

    ProfRecord *prof = ProfRecordManager::Get("simple");

    if (prof)
        prof->Dump(1);

 
    destroy_runtime_graph(graph);
    remove_model(model_name);

    std::cout << "ALL TEST DONE\n";

    return 0;
}
