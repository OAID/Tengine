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

#define DEF_MODEL "models/mobilenet_ssd.tflite"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"
#define DEF_LABEL "models/coco_labels_list.txt"

void get_input_data_ssd(const char* image_file, float* input_data, int img_h, int img_w)
{
    image im = imread(image_file);

    image resImg = resize_image(im, img_w, img_h);

    float mean = 127.5;

    for(int h = 0; h < img_h; h++)
        for(int w = 0; w < img_w; w++)
            for(int c = 0; c < 3; c++)
            {
                int src_index = c * img_h * img_w + h * img_w + w;
                int dst_index = h * img_w * 3 + w * 3 + c;
                input_data[dst_index] = 0.007843 * (resImg.data[src_index] - mean);
            }
}

void LoadLabelFile(std::vector<std::string>& result)
{
    std::ifstream labels(DEF_LABEL);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void post_process_ssd(tensor_t concat0, tensor_t concat1)
{
    // float *anchor_ptr = (float *)get_tensor_mem(input_anchors);

    // float *box_ptr = (float *)get_tensor_buffer(concat0);
    float* score_ptr = ( float* )get_tensor_buffer(concat1);
    /*
    float *detection_boxes = (float *)get_tensor_mem(output_detection_boxes);
    float *detection_classes = (float *)get_tensor_mem(output_detection_classes);
    float *detection_scores = (float *)get_tensor_mem(output_detection_scores);
    float *num_detections = (float *)get_tensor_mem(output_num_detections);
    */
    int dims[4] = {0};
    int dims_size = get_tensor_shape(concat0, dims, 4);
    std::cout << "box shape: [";
    for(int i = 0; i < dims_size; i++)
        std::cout << dims[i] << ",";
    std::cout << "]\n";

    int num_boxes = dims[1];
    int num_classes = 90;

    std::vector<std::string> labels;
    LoadLabelFile(labels);

    for(int j = 0; j < num_boxes; j++)
    {
        float max_score = 0.f;
        int class_idx = 0;
        for(int i = 0; i < num_classes; i++)
        {
            float score = score_ptr[j * (num_classes + 1) + i + 1];
            if(score > max_score)
            {
                max_score = score;
                class_idx = i + 1;
            }
        }
        if(max_score > 0.6)
        {
            std::cout << "score: " << max_score << " class: " << labels[class_idx] << "\n";
        }
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
    int img_h = 300;
    int img_w = 300;
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
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    // warm up
    get_input_data_ssd(DEF_IMAGE, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);
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

    tensor_t concat0 = get_graph_output_tensor(graph, 0, 0);
    tensor_t concat1 = get_graph_output_tensor(graph, 1, 0);

    post_process_ssd(concat0, concat1);

    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
    }
    free(input_data);

    destroy_graph(graph);

    return 0;
}
