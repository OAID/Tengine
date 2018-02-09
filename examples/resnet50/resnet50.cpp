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

#include "utils.hpp"

const char *label_file = "./synset_words.txt";

int main(int argc, char *argv[])
{
    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;

    if (argc < 3)
    {
        std::cout << "[usage]: " << argv[0] << " <test.jpg>  <model_dir> \n";
        return 0;
    }
    std::string img_name = argv[1];
    std::string model_dir = argv[2];

    const char *model_name = "resnet50";
    std::string proto_name = model_dir + "/resnet50.prototxt";
    std::string mdl_name = model_dir + "/resnet50.caffemodel";
    const char *proto_name_ = proto_name.c_str();
    const char *mdl_name_ = mdl_name.c_str();

    // load model
    const char *input_node_name = "input";
    if (load_model(model_name, "caffe", proto_name_, mdl_name_) < 0)
        return 1;
    std::cout << "load resnet50 model done!\n";

    // create graph
    graph_t graph = create_runtime_graph("resnet50_graph", model_name, NULL);
    set_graph_input_node(graph, &input_node_name, 1);

    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    // input
    int img_h = 224;
    int img_w = 224;
    int img_size = img_h * img_w * 3;
    float *input_data = (float *)malloc(sizeof(float) * img_size);
    get_input_data(img_name, input_data, img_h, img_w);

    const char *input_tensor_name = "data";
    tensor_t input_tensor = get_graph_tensor(graph, input_tensor_name);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        std::printf("set buffer for tensor: %s failed\n", input_tensor_name);
        return -1;
    }

    // run the graph
    prerun_graph(graph);

    for(int i=0;i<10;i++)
    run_graph(graph, 1);

    // unsigned long long t0,t1;
    // int repeat = 15;
    // float avg_time = 0.f;
    // for (int i = 0; i < repeat; i++)
    // {
    //     t0 = get_cur_time();
    //     run_graph(graph, 1);
    //     t1 = get_cur_time();
    //     unsigned long long mytime =(t1 - t0);
    //     std::cout << "i =" << i << " time is " << mytime << "us \n";
    //     avg_time += mytime;
    // }
    // std::cout<<"repeat "<<repeat<<", avg time is "<<avg_time/repeat/1000.f<<" ms \n ";

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
    destroy_runtime_graph(graph);
    remove_model(model_name);

    return 0;
}
