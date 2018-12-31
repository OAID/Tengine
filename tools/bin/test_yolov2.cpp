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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "operator/region.hpp"
#include "node.hpp"
#include <sys/time.h>

#define DEF_PROTO "models/yolo-voc.prototxt"
#define DEF_MODEL "models/yolo-voc.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

using namespace TEngine;

#define REPEAT_COUNT 1000

const char* image_list = "/home/firefly/my_tengine/tengine/tools/data/2007_test.txt";
const std::string root_path = "/home/firefly/my_tengine/tengine/";

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
struct GROUND_TRUTH
{
    Box box;
    int class_index;
    bool b_find;
};

struct MAP
{
    int tp_cnt;    // true positive
    int fp_cnt;    // false positive
    int tn_cnt;    // ture negitative
    int fn_cnt;    // false negetive
};

const char* class_names[] = {"background", "aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",
                             "car",        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
                             "motorbike",  "person",    "pottedplant", "sheep", "sofa",        "train",  "tvmonitor"};

void LoadImageFile(std::vector<std::string>& result, const char* fname)
{
    std::ifstream images(fname);

    std::string line;
    while(std::getline(images, line))
    {
        result.push_back(line);
    }
}

void string_replace(std::string& str, const std::string& srcstr, const std::string dststr)
{
    std::string::size_type pos = 0;
    std::string::size_type srclen = srcstr.size();
    std::string::size_type dstlen = dststr.size();

    while((pos = str.find(srcstr, pos)) != std::string::npos)
    {
        str.replace(pos, srclen, dststr);
        pos += dstlen;
    }
}

GROUND_TRUTH* read_boxes(char* filename, int* n)
{
    FILE* file = fopen(filename, "r");

    float x, y, h, w;
    int id = 0;
    int count = 0;
    int size = 64;

    GROUND_TRUTH* boxes = ( GROUND_TRUTH* )calloc(size, sizeof(GROUND_TRUTH));

    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5)
    {
        if(count == size)
        {
            size = size * 2;
            boxes = ( GROUND_TRUTH* )realloc(boxes, sizeof(GROUND_TRUTH));
        }
        boxes[count].class_index = id;
        boxes[count].box.x = x;
        boxes[count].box.y = y;
        boxes[count].box.w = w;
        boxes[count].box.h = h;
        boxes[count].b_find = false;
        count++;
    }
    fclose(file);
    *n = count;

    return boxes;
}
/******************************************
******   function_name: record_tp
******   image_file   : The image file of the test images
******   int num      : The detetion boxes num of the image
******   float thresh : The probs thresh(0.24)
******   std::vector<Box> &boxes : the detected boxes
******   float **probs:  the score value;
******   int class    :  21
******   GROUND_TRUTH :  the ground truth boxes
******   int ground_num:  the ground truth boxes num
******   MAP* result_map:  the reult map

*/

void record_tp(std::string& image_file, int num, float thresh, std::vector<Box>& boxes, float** probs, int classes,
               GROUND_TRUTH* ground_truth, int ground_num, MAP* result_map)
{
#if 0
    	const char *class_names[] = {"background",
		"aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor"};
#endif
    cv::Mat img = cv::imread(image_file);

    int i, j;
    float iou_thresh = 0.5;
    bool flag = false;
    int count = 0;

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
                for(int g = 0; g < ground_num; g++)
                {
                    if(ground_truth[g].class_index == class_id && ground_truth[g].b_find == false)
                    {
                        if(box_iou(ground_truth[g].box, boxes[i]) > iou_thresh)
                        {
                            flag = true;
                            ground_truth[g].b_find = true;
                            count++;
                        }
                    }
                }
                if(flag == true)
                {
                    std::cout << "tp id :" << class_id << "\n";
                    flag = false;
                    result_map[class_id].tp_cnt++;
                }
                else
                {
                    std::cout << "fp id :" << class_id << "\n";
                    result_map[class_id].fp_cnt++;
                }
            }
        }
    }
    if(count != ground_num)
    {
        int class_id = -1;
        for(int g = 0; g < ground_num; g++)
        {
            if(ground_truth[g].b_find == false)
            {
                class_id = ground_truth[g].class_index;

                std::cout << "fn id :" << class_id << "\n";

                result_map[class_id].fn_cnt++;
            }
        }
    }
}

void cal_recall_prob(MAP* map, int class_num)
{
    for(int i = 0; i < class_num; i++)
    {
        float recall = 0;
        float prob = 0;

        if(map[i].tp_cnt + map[i].fn_cnt == 0)
        {
            recall = 0;
        }
        else
        {
            recall = ( float )map[i].tp_cnt / (map[i].tp_cnt + map[i].fn_cnt);
        }

        if(map[i].tp_cnt + map[i].fp_cnt == 0)
        {
            prob = 0;
        }
        else
        {
            prob = ( float )map[i].tp_cnt / (map[i].tp_cnt + map[i].fp_cnt);
        }
        printf("class  %16s : recall: %4f, precesion: %4f\n", class_names[i + 1], recall, prob);

        // std::cout<<"class:   "<<class_names[i+1]<<":"<<"  recall:   "<<recall<<"   ,precesion:   "<<prob<<"\n";
    }
}
void preprocess_yolov2(std::string& image_file, float* input_data, int img_h, int img_w, int* raw_h, int* raw_w)
{
    cv::Mat img0 = cv::imread(image_file, -1);

    std::cout << "image_file :" << image_file << "\n";

    if(img0.empty())
    {
        std::cerr << "failed to read image file " << image_file << "\n";
        return;
    }

    *raw_h = img0.rows;
    *raw_w = img0.cols;

    int new_w = img0.cols;
    int new_h = img0.rows;
    if((( float )img_w / img0.cols) < (( float )img_h / img0.rows))
    {
        new_w = img_w;
        new_h = (img0.rows * img_w) / img0.cols;
    }
    else
    {
        new_h = img_h;
        new_w = (img0.cols * img_h) / img0.rows;
    }

    cv::Mat img;
    if(img0.channels() == 4)
    {
        cv::cvtColor(img0, img, cv::COLOR_BGRA2BGR);
    }
    else if(img0.channels() == 1)
    {
        cv::cvtColor(img0, img, cv::COLOR_GRAY2BGR);
    }
    else
    {
        img = img0;
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
    std::string proto_file;
    std::string model_file;
    std::string image_file;

    std::vector<std::string> images;

    LoadImageFile(images, image_list);

    // this thresh can be tuned for higher/lower confidence boxes
    float thresh = 0.24;
    int class_num = 21;

    MAP* global_map = ( MAP* )malloc(sizeof(struct MAP) * class_num);

    memset(global_map, 0, sizeof(MAP) * class_num);

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    // load model
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

    // create graph
    graph_t graph = create_graph(nullptr, "caffe", proto_file.c_str(), model_file.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    prerun_graph(graph);

    // input
    int img_h = 416;
    int img_w = 416;
    int raw_h = 0, raw_w = 0;
    int img_size = img_h * img_w * 3;
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);

    for(int i = 0; i < REPEAT_COUNT; i++)
    {
        image_file = images[i];
        std::cout << "=================== " << i << " ========================\n";

        preprocess_yolov2(image_file, input_data, img_h, img_w, &raw_h, &raw_w);

        set_tensor_buffer(input_tensor, input_data, 3 * img_h * img_w * 4);

        run_graph(graph, 1);

        tensor_t tensor = get_graph_output_tensor(graph, 0, 0);
        int out_dim[4] = {0};
        get_tensor_shape(tensor, out_dim, 4);
        float* output = ( float* )get_tensor_buffer(tensor);

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

        // char *ground_truth_file = "/home/firefly/VOCdevkit/VOC2007/labels/000001.txt";

        std::string src_sub = "JPEGImages";
        std::string dst_sub = "labels";
        std::string src_type = "jpg";
        std::string dst_type = "txt";

        std::string ground_truth_file = image_file;

        string_replace(ground_truth_file, src_sub, dst_sub);
        string_replace(ground_truth_file, src_type, dst_type);

        int ground_num = 0;
        GROUND_TRUTH* ground_truth = read_boxes(( char* )ground_truth_file.c_str(), &ground_num);

        std::cout << "ground num is :" << ground_num << "\n";

        record_tp(image_file, total, thresh, boxes, probs, num_class, ground_truth, ground_num, global_map);

        free(ground_truth);
        for(int j = 0; j < total; ++j)
        {
            free(probs[j]);
        }
        free(probs);
    }

    cal_recall_prob(global_map, 20);

    free(global_map);
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
