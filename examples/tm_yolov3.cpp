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
 * Copyright (c) 2021, OPEN AI LAB
 */

#include <iostream>
#include <iomanip>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

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
const float thresh = 0.55;
const float hier_thresh = 0.5;
const float nms = 0.45;
const int relative = 1;

// yolov3
float biases[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
// tiny
float biases_tiny[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
// yolov2
float biases_yolov2[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes, int layer_type)
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
    if (layer_type == 0)
    {
        l.mask = ( int* )calloc(n, sizeof(int));
        if (9 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 6;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 8)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
        if (6 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases_tiny[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
    }
    else if (1 == layer_type)
    {
        l.coords = 4;
        for (int i = 0; i < total * 2; ++i)
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
    if (NULL != l.biases)
    {
        free(l.biases);
        l.biases = NULL;
    }
    if (NULL != l.mask)
    {
        free(l.mask);
        l.mask = NULL;
    }
    if (NULL != l.output)
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
    for (int i = 0; i < size; ++i)
    {
        input[i] = 1.f / (1.f + expf(-input[i]));
    }
}

void forward_darknet_layer_cpu(const float* input, layer l)
{
    memcpy(( void* )l.output, ( void* )input, sizeof(float) * l.inputs * l.batch);
    if (0 == l.layer_type)
    {
        for (int b = 0; b < l.batch; ++b)
        {
            for (int n = 0; n < l.n; ++n)
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
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                if (l.output[obj_index] > thresh)
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
    for (i = 0; i < ( int )layers_params.size(); ++i)
    {
        layer l = layers_params[i];
        if (0 == l.layer_type)
            s += yolo_num_detections(l, thresh);
        else if (1 == l.layer_type)
            s += l.w * l.h * l.n;
    }

    fprintf(stderr, "%s,%d\n", __func__, s);
    return s;
}

detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params, thresh);
    if (num)
        *num = nboxes;
    detection* dets = ( detection* )calloc(nboxes, sizeof(detection));

    for (i = 0; i < nboxes; ++i)
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
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
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
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            int row = i / l.w;
            int col = i % l.w;
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                float objectness = predictions[obj_index];
                if (objectness <= thresh)
                    continue;
                int box_index = entry_index(l, b, n * l.w * l.h + i, 0);

                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw,
                                                neth, l.w * l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j)
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

void correct_region_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_region_box(float* x, float* biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, float tree_thresh,
                           int relative, detection* dets)
{
    int i, j, n;
    float* predictions = l.output;

    for (i = 0; i < l.w * l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n)
        {
            int index = n * l.w * l.h + i;
            for (j = 0; j < l.classes; ++j)
            {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n * l.w * l.h + i, l.coords);
            int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
            int mask_index = entry_index(l, 0, n * l.w * l.h + i, 4);
            float scale = predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w * l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask)
            {
                for (j = 0; j < l.coords - 4; ++j)
                {
                    dets[index].mask[j] = l.output[mask_index + j * l.w * l.h];
                }
            }
            // int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1);
            if (dets[index].objectness)
            {
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, 0, n * l.w * l.h + i, l.coords + 1 + j);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, l.w * l.h * l.n, w, h, netw, neth, relative);
}

void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                        float hier, int* map, int relative, detection* dets)
{
    int j;
    for (j = 0; j < ( int )layers_params.size(); ++j)
    {
        layer l = layers_params[j];
        if (0 == l.layer_type)
        {
            int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
            dets += count;
        }
        else
        {
            get_region_detections(l, img_w, img_h, net_w, net_h, thresh, map, hier, relative, dets);
            dets += l.w * l.h * l.n;
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
    for (i = 0; i < nboxes; ++i)
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
    if (b.sort_class >= 0)
    {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else
    {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0)
        return 1;
    else if (diff > 0)
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
    if (w < 0 || h < 0)
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
    for (i = 0; i <= k; ++i)
    {
        if (dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0)
                continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

image letterbox_image(image im, int w, int h);

void rgbgr_image(image im)
{
    int i;
    for (i = 0; i < im.w * im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w * im.h * 2];
        im.data[i + im.w * im.h * 2] = swap;
    }
}

void fill_image(image m, float s)
{
    int i;
    for (i = 0; i < m.h * m.w * m.c; ++i)
        m.data[i] = s;
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if ((( float )w / im.w) < (( float )h / im.h))
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

void get_input_data_darknet(const char* image_file, float* input_data, int net_h, int net_w)
{
    int size = 3 * net_w * net_h;
    image sized;
    image im = load_image_stb(image_file, 3);
    for (int i = 0; i < im.c * im.h * im.w; i++)
    {
        im.data[i] = im.data[i] / 255;
    }
    sized = letterbox(im, net_w, net_h);
    memcpy(input_data, sized.data, size * sizeof(float));

    free_image(sized);
    free_image(im);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;

    int layer_type = 0;
    int numBBoxes = 3;
    int total_numAnchors = 9;
    int net_w = 608;
    int net_h = 608;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, nullptr, 10);
                break;
            case 't':
                num_thread = std::strtoul(optarg, nullptr, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = net_h * net_w * 3;
    int dims[] = {1, 3, net_h, net_w};    // nchw

    std::vector<float> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_darknet(image_file, input_data.data(), net_h, net_w);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    image img = imread(image_file);
    int output_node_num = get_graph_output_node_number(graph);

    vector<layer> layers_params;
    layers_params.clear();
    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
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
        get_network_boxes(layers_params, img.w, img.h, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);

    if (nms != 0)
    {
        do_nms_sort(dets, nboxes, classes, nms);
    }

    int i, j;
    for (i = 0; i < nboxes; ++i)
    {
        int cls = -1;
        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j] > 0.5)
            {
                if (cls < 0)
                {
                    cls = j;
                }
                fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
            }
        }
        if (cls >= 0)
        {
            box b = dets[i].bbox;
            int left = (b.x - b.w / 2.) * img.w;
            int right = (b.x + b.w / 2.) * img.w;
            int top = (b.y - b.h / 2.) * img.h;
            int bot = (b.y + b.h / 2.) * img.h;
            draw_box(img, left, top, right, bot, 2, 125, 0, 125);
            fprintf(stderr, "left = %d,right = %d,top = %d,bot = %d\n", left, right, top, bot);
        }

        if (dets[i].mask)
            free(dets[i].mask);
        if (dets[i].prob)
            free(dets[i].prob);
    }
    free(dets);
    save_image(img, "tengine_example_out");

    /* release tengine */
    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
        release_graph_tensor(out_tensor);
    }

    free_image(img);

    for (int i = 0; i < layers_params.size(); i++)
    {
        layer l = layers_params[i];
        if (l.output)
            free(l.output);
        if (l.biases)
            free(l.biases);
        if (l.mask)
            free(l.mask);
    }

    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
