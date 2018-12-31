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
#include <iomanip>
#include <string>
#include <vector>
#include <typeinfo>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "common.hpp"
#include <sys/time.h>
#include "common.hpp"

#define DEF_PROTO "models/yolo-voc.prototxt"
#define DEF_MODEL "models/yolo-voc.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

struct Box
{
    float x;
    float y;
    float w;
    float h;
};

struct Sbox
{
    int index;
    int class_id;
    float** probs;
};

static int nms_comparator(const void* pa, const void* pb)
{
    Sbox a = *( Sbox* )pa;
    Sbox b = *( Sbox* )pb;
    float diff = a.probs[a.index][b.class_id] - b.probs[b.index][b.class_id];
    if(diff < 0)
        return 1;
    else if(diff > 0)
        return -1;
    return 0;
}

int entry_index(int n, int loc, int entry, int hw, int classes)
{
    int coords = 4;
    return n * hw * (coords + classes + 1) + entry * hw + loc;
}

void get_region_box(Box& b, float* x, std::vector<float>& biases, int n, int index, int i, int j, int w, int h,
                    int stride)
{
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
}

void correct_region_boxes(std::vector<Box>& boxes, int n, int w, int h, int netw, int neth)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for(i = 0; i < n; ++i)
    {
        Box b = boxes[i];
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        boxes[i] = b;
    }
}

void get_region_boxes(float* output, std::vector<float>& biases, int neth, int netw, int h, int w, int img_w, int img_h,
                      int num_box, int num_classes, float thresh, float** probs, std::vector<Box>& boxes)
{
    int coords = 4;
    int hw = h * w;
    int i, j, n;
    float* predictions = output;

    for(i = 0; i < hw; ++i)
    {
        int row = i / w;
        int col = i % w;
        for(n = 0; n < num_box; ++n)
        {
            int index = n * hw + i;
            for(j = 0; j < num_classes; ++j)
            {
                probs[index][j] = 0;
            }
            int obj_index = entry_index(n, i, coords, hw, num_classes);
            int box_index = entry_index(n, i, 0, hw, num_classes);
            float scale = predictions[obj_index];
            get_region_box(boxes[index], predictions, biases, n, box_index, col, row, w, h, hw);

            float max = 0;
            for(j = 0; j < num_classes; ++j)
            {
                int class_index = entry_index(n, i, coords + 1 + j, hw, num_classes);
                float prob = scale * predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max)
                    max = prob;
            }
            probs[index][num_classes] = max;
        }
    }

    correct_region_boxes(boxes, hw * num_box, img_w, img_h, netw, neth);
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Box& a, Box& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0)
        return 0;
    float area = w * h;
    return area;
}

float box_union(Box& a, Box& b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(Box& a, Box& b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(std::vector<Box>& boxes, float** probs, int total, int classes, float thresh)
{
    int i, j, k;
    Sbox* s = ( Sbox* )malloc(sizeof(Sbox) * total);

    for(i = 0; i < total; ++i)
    {
        s[i].index = i;
        s[i].class_id = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k)
    {
        for(i = 0; i < total; ++i)
        {
            s[i].class_id = k;
        }
        qsort(s, total, sizeof(Sbox), nms_comparator);
        for(i = 0; i < total; ++i)
        {
            if(probs[s[i].index][k] == 0)
                continue;
            Box a = boxes[s[i].index];
            for(j = i + 1; j < total; ++j)
            {
                Box b = boxes[s[j].index];
                if(box_iou(a, b) > thresh)
                {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

void draw_detections(std::string& image_file, std::string& save_name, int num, float thresh, std::vector<Box>& boxes,
                     float** probs, int classes)
{
    const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};
    cv::Mat img = cv::imread(image_file);
    int img_h = img.size().height;
    int img_w = img.size().width;
    int line_width = img_w * 0.005;
    int i, j;
    for(i = 0; i < num; ++i)
    {
        int class_id = -1;
        for(j = 0; j < classes; ++j)
        {
            if(probs[i][j] > thresh)
            {
                if(class_id < 0)
                {
                    class_id = j;
                }
                printf("%s\t:%.0f%%\n", class_names[class_id + 1], probs[i][j] * 100);
                Box b = boxes[i];
                int left = (b.x - b.w / 2.) * img_w;
                int right = (b.x + b.w / 2.) * img_w;
                int top = (b.y - b.h / 2.) * img_h;
                int bot = (b.y + b.h / 2.) * img_h;
                if(left < 0)
                    left = 0;
                if(right > img_w - 1)
                    right = img_w - 1;
                if(top < 0)
                    top = 0;
                if(bot > img_h - 1)
                    bot = img_h - 1;
                printf("BOX:( %d , %d ),( %d , %d )\n", left, top, right, bot);
                cv::rectangle(img, cv::Rect(left, top, (right - left), (bot - top)), cv::Scalar(0, 255, 255),
                              line_width);
                std::ostringstream score_str;
                score_str << probs[i][j];
                std::string label = std::string(class_names[class_id + 1]) + ": " + score_str.str();
                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(img,
                              cv::Rect(cv::Point(left, top - label_size.height),
                                       cv::Size(label_size.width, label_size.height + baseLine)),
                              cv::Scalar(0, 255, 255), CV_FILLED);
                cv::putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
    }

    cv::imwrite(save_name, img);
    std::cout << "======================================\n";
    std::cout << "[DETECTED IMAGE SAVED]:\t" << save_name << "\n";
    std::cout << "======================================\n";
}

void preprocess_yolov2(std::string& image_file, float* input_data, int img_h, int img_w, int* raw_h, int* raw_w)
{
    cv::Mat img = cv::imread(image_file, -1);
    if(img.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }

    *raw_h = img.rows;
    *raw_w = img.cols;

    int new_w = img.cols;
    int new_h = img.rows;
    if((( float )img_w / img.cols) < (( float )img_h / img.rows))
    {
        new_w = img_w;
        new_h = (img.rows * img_w) / img.cols;
    }
    else
    {
        new_h = img_h;
        new_w = (img.cols * img_h) / img.rows;
    }

    img.convertTo(img, CV_32FC3);
    img = img.mul(0.00392156862745098f);

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat temp = channels[2];
    channels[2] = channels[0];
    channels[0] = temp;
    cv::merge(channels, img);
    cv::Mat resize_img;
    cv::Mat dst_img;
    cv::resize(img, resize_img, cv::Size(new_w, new_h));

    int delta_h = (img_h - new_h) * 0.5f;
    int delta_w = (img_w - new_w) * 0.5f;
    cv::copyMakeBorder(resize_img, dst_img, delta_h, delta_h, delta_w, delta_w, cv::BORDER_CONSTANT, cv::Scalar(0.5f));

    float* img_data = ( float* )dst_img.data;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = *img_data;
                img_data++;
            }
        }
    }
}

int main(int argc, char** argv)
{
    int ret = -1;
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    std::string image_file;
    std::string save_name = "save.jpg";

    // this thresh can be tuned for higher/lower confidence boxes
    float thresh = 0.24;

    int res;
    while((res = getopt(argc, argv, "p:m:i:")) != -1)
    {
        switch(res)
        {
            case 'p':
                proto_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            default:
                break;
        }
    }

    // init tengine
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    if(request_tengine_version("0.9") < 0)
    {
        std::cout << " request tengine version failed\n";
        return 1;
    }

    // load model &  create graph
    if(proto_file.empty())
    {
        proto_file = root_path + DEF_PROTO;
        std::cout << "proto file not specified,using " << proto_file << " by default\n";
    }
    if(model_file.empty())
    {
        model_file = root_path + DEF_MODEL;
        std::cout << "model file not specified,using " << model_file << " by default\n";
    }
    if(image_file.empty())
    {
        image_file = root_path + DEF_IMAGE;
        std::cout << "image file not specified,using " << image_file << " by default\n";
    }
    // check file
    if(!check_file_exist(proto_file) or (!check_file_exist(model_file) or !check_file_exist(image_file)))
    {
        return 1;
    }
    graph_t graph = create_graph(nullptr, "caffe", proto_file.c_str(), model_file.c_str());

    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // input
    int img_h = 416;
    int img_w = 416;
    int raw_h = 0, raw_w = 0;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4);

    // prerun
    ret = prerun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // output
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    int out_dim[4] = {0};
    get_tensor_shape(output_tensor, out_dim, 4);

    float* output = ( float* )get_tensor_buffer(output_tensor);

    int repeat_count = 1;
    const char* repeat = std::getenv("REPEAT_COUNT");

    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float avg_time = 0.f;
    for(int i = 0; i < repeat_count; i++)
    {
        // image_file="ssd_horse.jpg";
        // save_name="out/"+std::to_string(i)+".jpg";

        preprocess_yolov2(image_file, input_data, img_h, img_w, &raw_h, &raw_w);

        gettimeofday(&t0, NULL);

        ret = run_graph(graph, 1);
        if(ret != 0)
        {
            std::cout << "Run graph failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }

        gettimeofday(&t1, NULL);
        float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        avg_time += mytime;

        node_t node = get_graph_node(graph, "region");
        int num_box = 0;
        int num_class = 0;

        if(get_node_attr_int(node, "num_box", &num_box) < 0)
        {
            std::cerr << "cannot get num box setting\n";
            return 1;
        }

        if(get_node_attr_int(node, "num_classes", &num_class) < 0)
        {
            std::cerr << "cannot get num class setting\n";
            return 1;
        }

        std::vector<float> param_biases;

        if(get_node_attr_generic(node, "biases", &typeid(std::vector<float>), &param_biases, sizeof(param_biases)) < 0)
        {
            std::cout << "cannot get bias settings\n";
            return 1;
        }

        // printf("num box: %d\n",num_box);
        // printf("num class: %d\n",num_class);

        int total = out_dim[2] * out_dim[3] * num_box;
        // init box and probs
        std::vector<Box> boxes(total);
        float** probs = ( float** )calloc(total, sizeof(float*));
        for(int j = 0; j < total; ++j)
        {
            probs[j] = ( float* )calloc(num_class + 1, sizeof(float*));
        }

        get_region_boxes(output, param_biases, img_h, img_w, out_dim[2], out_dim[3], raw_w, raw_h, num_box, num_class,
                         thresh, probs, boxes);

        float nms_thresh = 0.3;
        do_nms_sort(boxes, probs, total, num_class, nms_thresh);
        // if repeat_count=1, print output
        if(i == repeat_count - 1)
            draw_detections(image_file, save_name, total, thresh, boxes, probs, num_class);

        for(int j = 0; j < total; ++j)
            free(probs[j]);
        free(probs);
    }
    std::cout << "--------------------------------------\n";
    std::cout << "repeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";

    // free
    free(input_data);
    release_graph_tensor(output_tensor);
    release_graph_tensor(input_tensor);
    ret = postrun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    destroy_graph(graph);
    release_tengine();
    return 0;
}
