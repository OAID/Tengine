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
 * Author: ruizhang@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>
#include <stdlib.h>
#include <algorithm>

#include "tengine_c_api.h"
#include "tengine_operations.h"
#include <math.h>

using namespace std;

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
} detection;

typedef struct layer
{
    int layer_type;
    int batch;
    int total;
    int n, c, h, w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs;
    int outputs;
    int* mask;
    float* biases;
    float* output;
    int coords;
} layer;

const int classes = 80;
const float thresh = 0.5;
const float hier_thresh = 0.5;
const float nms = 0.45;
const int numBBoxes = 5;
const int relative = 1;
const int yolov3_numAnchors = 6;
const int yolov2_numAnchors = 5;

//yolov3
float biases[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
// tiny
float biases_tiny[12] = {10,14,  23,27,  37,58 , 81,82,  135,169,  344,319};
//yolov2
float biases_yolov2[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes,int layer_type)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w * l.h * l.c;

    l.biases = ( float* )calloc(total * 2, sizeof(float));
    if(layer_type == 0)
    {
        l.mask = ( int* )calloc(n, sizeof(int));
        if(9 == total)
        {
            for(int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases[i];
            }
            if(l.w == net_w / 32)
            {
                int j = 6;
                for(int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if(l.w == net_w / 16)
            {
                int j = 3;
                for(int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if(l.w == net_w / 8)
            {
                int j = 0;
                for(int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
        if(6 == total){
            for(int i =0;i<total*2;++i){

                l.biases[i] = biases_tiny[i];
            }
            if(l.w == net_w / 32){

                int j = 3;

                for(int i =0;i<l.n;++i)

                    l.mask[i] = j++;

            }

            if(l.w == net_w / 16){

                int j = 0;

                for(int i =0;i<l.n;++i)

                    l.mask[i] = j++;

            }

        }
    }
    else if(1 == layer_type)
    {
        l.coords = 4;
        for(int i =0;i<total*2;++i)
        {
            l.biases[i] = biases_yolov2[i];
        }
    }
    l.layer_type = layer_type;
    l.outputs = l.inputs;
    l.output = ( float* )calloc(batch * l.outputs, sizeof(float));

    return l;
}

void free_darknet_layer(layer l)
{
    if(NULL != l.biases)
    {
        free(l.biases);
        l.biases = NULL;
    }
    if(NULL != l.mask)
    {
        free(l.mask);
        l.mask = NULL;
    }
    if(NULL != l.output)
    {
        free(l.output);
        l.output = NULL;
    }
}
static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}
void logistic_cpu(float* input, int size)
{
    for(int i = 0; i < size; ++i)
    {
        input[i] = 1.f / (1.f + expf(-input[i]));
    }
}

void forward_darknet_layer_cpu(const float* input, layer l)
{
    memcpy(( void* )l.output, ( void* )input, sizeof(float) * l.inputs * l.batch);
    if(0 == l.layer_type)
    {
        for(int b = 0; b < l.batch; ++b)
        {
            for(int n = 0; n < l.n; ++n)
            {
                int index = entry_index(l, b, n * l.w * l.h, 0);
                logistic_cpu(l.output + index, 2 * l.w * l.h);
                index = entry_index(l, b, n * l.w * l.h, 4);
                logistic_cpu(l.output + index, (1 + l.classes) * l.w * l.h);
            }
        }
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n, b;
    int count = 0;
    for(b = 0; b < l.batch; ++b)
    {
        for(i = 0; i < l.w * l.h; ++i)
        {
            for(n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                if(l.output[obj_index] > thresh)
                    ++count;
            }
        }
    }
    return count;
}
int num_detections(vector<layer> layers_params, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < ( int )layers_params.size(); ++i)
    {
        layer l = layers_params[i];
        if(0 == l.layer_type)
            s += yolo_num_detections(l, thresh);
        else if(1 == l.layer_type)
            s += l.w*l.h*l.n;
    }


    printf("%s,%d\n",__func__,s);
    return s;
}
detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params, thresh);
    if(num)
        *num = nboxes;
    detection* dets = ( detection* )calloc(nboxes, sizeof(detection));

    for(i = 0; i < nboxes; ++i)
    {
        dets[i].prob = ( float* )calloc(l.classes, sizeof(float));
    }
    return dets;
}

void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
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
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if(!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
box get_yolo_box(float* x, float* biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;

    return b;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, int relative,
                        detection* dets)
{
    int i, j, n, b;
    float* predictions = l.output;
    int count = 0;
    for(b = 0; b < l.batch; ++b)
    {
        for(i = 0; i < l.w * l.h; ++i)
        {
            int row = i / l.w;
            int col = i % l.w;
            for(n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                float objectness = predictions[obj_index];
                if(objectness <= thresh)
                    continue;
                int box_index = entry_index(l, b, n * l.w * l.h + i, 0);

                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw,
                                                neth, l.w * l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for(j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, b, n * l.w * l.h + i, 4 + 1 + j);
                    float prob = objectness * predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}
void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;

    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = n*l.w*l.h + i;
            for(j = 0; j < l.classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l.coords - 4; ++j){
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }
            //int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1);
            if(dets[index].objectness){
                for(j = 0; j < l.classes; ++j){
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                    float prob = scale*predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}

void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                        float hier, int* map, int relative, detection* dets)
{
    int j;
    for(j = 0; j < ( int )layers_params.size(); ++j)
    {
        layer l = layers_params[j];
        if(0 == l.layer_type)
        {
            int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
            dets += count;
        }
        else
        {
            get_region_detections(l, img_w,img_h, net_w, net_h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
    }

}

detection* get_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                             float hier, int* map, int relative, int* num)
{
    // make network boxes
    detection* dets = make_network_boxes(layers_params, thresh, num);

    // fill network boxes
    fill_network_boxes(layers_params, img_w, img_h, net_w, net_h, thresh, hier, map, relative, dets);
    return dets;
}
// release detection memory
void free_detections(detection* dets, int nboxes)
{
    int i;
    for(i = 0; i < nboxes; ++i)
    {
        free(dets[i].prob);
    }
    free(dets);
}

int nms_comparator(const void* pa, const void* pb)
{
    detection a = *( detection* )pa;
    detection b = *( detection* )pb;
    float diff = 0;
    if(b.sort_class >= 0)
    {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else
    {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0)
        return 1;
    else if(diff > 0)
        return -1;
    return 0;
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

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0)
        return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for(i = 0; i <= k; ++i)
    {
        if(dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for(k = 0; k < classes; ++k)
    {
        for(i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i)
        {
            if(dets[i].prob[k] == 0)
                continue;
            box a = dets[i].bbox;
            for(j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if(box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void free_image(image m);
image letterbox_image(image im, int w, int h);

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w * im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w * im.h * 2];
        im.data[i + im.w * im.h * 2] = swap;
    }
}

void free_image(image m)
{
    if(m.data)
    {
        free(m.data);
    }
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h * m.w * m.c; ++i)
        m.data[i] = s;
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if((( float )w / im.w) < (( float )h / im.h))
    {
        new_w = w;
        new_h = (im.h * w) / im.w;
    }
    else
    {
        new_h = h;
        new_w = (im.w * h) / im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    add_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
    free_image(resized);
    return boxed;
}


int main(int argc, char* argv[])
{
    std::string image_file = "./tests/images/ssd_dog.jpg";
    int layer_type = 0;
    std::string model_name = "yolov3";
    std::string input_file = "darknet_yolov3.txt";
    std::string text_file = "./models/yolov3.cfg";
    std::string model_file = "./models/yolov3.weights";
    std::string out_img_name = "darknet_yolo3_test";
    int numBBoxes = 3;
    int total_numAnchors = 9;
    int net_w = 608;
    int net_h = 608;
    int repeat_count = 1;
    int res;
    while((res = getopt(argc, argv, "n:p:m:l:i:r")) != -1)
    {
        switch(res)
        {
            case 'n':
                model_name = optarg;
                break;
            case 'p':
                text_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }
    if(model_name == "yolov2")
    {
        layer_type = 1;
        numBBoxes = 5;
        total_numAnchors = 5;
        input_file = "darknet_yolov2.txt";
        out_img_name = "darknet_yolo2_test";
    }
    else if(model_name == "yolov3_tiny")
    {
        layer_type = 0;
        numBBoxes = 3;
        total_numAnchors = 6;
        net_w = 416;
        net_h = 416;
        input_file = "darknet_yolov3_tiny.txt";
        out_img_name = "darknet_yolo3_tiny_test";
    }
    else if(model_name == "yolov2_tiny")
    {
        layer_type = 1;
        numBBoxes = 5;
        total_numAnchors = 5;
        net_w = 416;
        net_h = 416;
        input_file = "darknet_yolov2_tiny.txt";
        out_img_name = "darknet_yolov2_tiny_test";

    }
    float* input_data = ( float* )malloc(sizeof(float) * net_w * net_h * 3);
    int size = 3 * net_w * net_h;
    image sized;
    image im = load_image_stb(image_file.c_str(),3);
    for(int i = 0; i < im.c * im.h * im.w; ++i)
    {
        im.data[i] = im.data[i] / 255;
    }
    sized = letterbox(im, net_w, net_h);
    memcpy(input_data, sized.data, size * sizeof(float));
    int img_h = im.h;
    int img_w = im.w;

    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;
    graph_t graph = create_graph(nullptr, "darknet", text_file.c_str(), model_file.c_str());
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
    int dims[] = {1, 3, net_h, net_w};
    set_tensor_shape(input_tensor, dims, 4);
    /* setup input buffer */
    if(set_tensor_buffer(input_tensor, input_data, 3 * net_h * net_w * 4) < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }
    prerun_graph(graph);
    // time run_graph
    const char* repeat = std::getenv("REPEAT_COUNT");
    if(repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    struct timeval t0, t1;
    float avg_time = 0.f;
    gettimeofday(&t0, NULL);
    for(int i = 0; i < repeat_count; i++)
        run_graph(graph, 1);
    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
    avg_time += mytime;
    std::cout << "--------------------------------------\n";
    std::cout << "repeat " << repeat_count << " times, avg time per run is " << avg_time / repeat_count << " ms\n";

    // post process
    int output_node_num = get_graph_output_node_number(graph);

    vector<layer> layers_params;
    layers_params.clear();
    for(int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out");
        int out_dim[4];
        get_tensor_shape(out_tensor, out_dim, 4);
        layer l_params;
        int out_w = out_dim[3];
        int out_h = out_dim[2];
        l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes, layer_type);
        layers_params.push_back(l_params);
        float* out_data = ( float* )get_tensor_buffer(out_tensor);
        forward_darknet_layer_cpu(out_data, l_params);
    }
    int nboxes = 0;
    // get network boxes
    detection* dets =
        get_network_boxes(layers_params, img_w, img_h, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);
    // release layer memory
    for(int index = 0; index < ( int )layers_params.size(); ++index)
    {
        free_darknet_layer(layers_params[index]);
    }
    if(nms)
    {
        do_nms_sort(dets, nboxes, classes, nms);
    }
    image img = imread(image_file.c_str());
    int i, j;
    for(i = 0; i < nboxes; ++i)
    {
        int cls = -1;
        for(j = 0; j < classes; ++j)
        {
            if(dets[i].prob[j] > 0.5)
            {
                if(cls < 0)
                {
                    cls = j;
                }
                printf("%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
            }
        }
        if(cls >= 0)
        {
            box b = dets[i].bbox;
            printf("x = %f,y =  %f,w = %f,h =  %f\n", b.x, b.y, b.w, b.h);
            int left = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top = (b.y - b.h / 2.) * im.h;
            int bot = (b.y + b.h / 2.) * im.h;
            draw_box(img, left, top, right, bot, 2, 125, 0, 125);
            printf("left = %d,right =  %d,top = %d,bot =  %d\n", left, right, top, bot);
        }
    }
    save_image(img, "dk_yolo.jpg");

    // free
    for(int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out");
        release_graph_tensor(out_tensor);
    }
    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);
    release_tengine();

    return 0;
}
