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
 * Copyright (c) 2018, OPEN AI LAB
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include "tengine_operations.h"
#include "tengine_c_api.h"

#define DEF_MODEL "models/detect_tflite.tmfile"
#define DEF_IMAGE "images/ssd_dog.jpg"
#define DEF_LABEL "models/coco_labels_list.txt"
#define DEFAULT_REPEAT_CNT 1

void get_input_data_ssd(const char* image_file, uint8_t* input_data, int img_h, int img_w)
{
    image im = imread(image_file);

    image resImg = resize_image(im, img_w, img_h);

    int index = 0;
    for(int c = 0; c < 3; c++)
        for(int h = 0; h < img_h; h++)
            for(int w = 0; w < img_w; w++)
                input_data[index++] = (uint8_t )resImg.data[c * img_h * img_w + h * img_w + w];

    free_image(im);
    free_image(resImg);    
}

void LoadLabelFile(std::vector<std::string>& result)
{
    std::ifstream labels(DEF_LABEL);

    std::string line;
    while(std::getline(labels, line))
        result.push_back(line);
}

void post_process_ssd(tensor_t t0, tensor_t t1, tensor_t t2, tensor_t t3)
{
    image im = imread(DEF_IMAGE);
    float* boxes = ( float* )get_tensor_buffer(t0);
    float* classes = ( float* )get_tensor_buffer(t1);
    float* scores = ( float* )get_tensor_buffer(t2);
    float* num = ( float* )get_tensor_buffer(t3);

    std::vector<std::string> labels;
    LoadLabelFile(labels);

    int max_num = num[0];
    printf("detect num : %d\n", max_num);
    for(int i = 0; i < max_num; i++)
    {
        if(scores[i] > 0.6)
        {
            int class_idx = classes[i];
            printf("score: %f, class: %d, %s \n", scores[i], class_idx, labels[class_idx].c_str());
            printf("\t %d, %d, %d, %d\n", (int)boxes[i * 4] * (int)im.w, (int)boxes[i * 4 + 1] * (int)im.h, (int)boxes[i * 4 + 2] * (int)im.w,
                   (int)boxes[i * 4 + 3] * (int)im.h);
            draw_box(im, boxes[i * 4] * im.w, boxes[i * 4 + 1] * im.h, boxes[i * 4 + 2] * im.w, boxes[i * 4 + 3] * im.h,
                     2, 125, 0, 125);
        }
    }
    save_image(im, "tengine_example_out");
    free_image(im);
}

int main(int argc, char* argv[])
{
    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(0, "tengine", DEF_MODEL);
    if(graph == nullptr)
    {
        std::cout << "create graph failed!\n";
        return 1;
    }

    int repeat_count = DEFAULT_REPEAT_CNT;
    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    uint8_t* input_data = (uint8_t* )malloc(sizeof(uint8_t) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(NULL == input_tensor)
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);
        return 1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    /* prerun the graph */
    struct options opt;
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    if(std::getenv("NumThreadLite"))
        opt.num_thread = atoi(std::getenv("NumThreadLite"));
    if(std::getenv("NumClusterLite"))
        opt.cluster = atoi(std::getenv("NumClusterLite"));
    if(std::getenv("DataPrecision"))
        opt.precision = atoi(std::getenv("DataPrecision"));
    if(std::getenv("REPEAT"))
        repeat_count = atoi(std::getenv("REPEAT"));
    
    std::cout<<"Number Thread  : [" << opt.num_thread <<"], use export NumThreadLite=1/2/4 set\n";
    std::cout<<"CPU Cluster    : [" << opt.cluster <<"], use export NumClusterLite=0/1/2/3 set\n";
    std::cout<<"Data Precision : [" << opt.precision <<"], use export DataPrecision=0/1/2/3 set\n";
    std::cout<<"Number Repeat  : [" << repeat_count <<"], use export REPEAT=10/100/1000 set\n";

    if(prerun_graph_multithread(graph, opt) < 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    get_input_data_ssd(DEF_IMAGE, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size);
    
    struct timeval t0, t1;
    float total_time = 0.f;
    float min_time = __DBL_MAX__;
    float max_time = -__DBL_MAX__;
    for(int i = 0; i < repeat_count; i++)
    {
        gettimeofday(&t0, NULL);
        run_graph(graph, 1);

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        total_time += mytime;
        min_time = std::min(min_time, mytime);
        max_time = std::max(max_time, mytime);
    }
    std::cout << "--------------------------------------\n";
    std::cout << "\nRepeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n" << "max time is " << max_time << " ms, min time is " << min_time << " ms\n";

    tensor_t boxes = get_graph_output_tensor(graph, 0, 0);
    tensor_t classes = get_graph_output_tensor(graph, 0, 1);
    tensor_t scores = get_graph_output_tensor(graph, 0, 2);
    tensor_t number = get_graph_output_tensor(graph, 0, 3);

    post_process_ssd(boxes, classes, scores, number);

    if(postrun_graph(graph) != 0)
    {
        std::cout << "postrun graph failed\n";
        return -1;
    }
    free(input_data);

    destroy_graph(graph);

    return 0;
}
