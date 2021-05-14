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

#include <cstdio>

#include <cstdlib>
#include <cstdio>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#endif
#include <algorithm>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

float s_anchors[] = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};

typedef struct layer
{
    int total_anchor;
    int box, c, h, w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs;
    int outputs;
    int* anchor_mask;
    float* anchors;
    float* output;
    int coords;
} layer;

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    float x, y, w, h;
    int classes;
    float* prob;
    float objectness;
    int sort_class;
} detection;

layer make_darknet_layer(int w, int h, int net_w, int net_h, int n, int total, int classes)
{
    layer l = {0};
    l.box = n;
    l.total_anchor = total;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w * l.h * l.c;

    l.anchors = ( float* )calloc(total * 2, sizeof(float));
    l.anchor_mask = ( int* )calloc(n, sizeof(int));
    if (9 == total)
    {
        for (int i = 0; i < total * 2; ++i)
        {
            l.anchors[i] = s_anchors[i];
        }
        if (l.w == net_w / 32)
        {
            int j = 6;
            for (int i = 0; i < l.box; ++i)
                l.anchor_mask[i] = j++;
        }
        if (l.w == net_w / 16)
        {
            int j = 3;
            for (int i = 0; i < l.box; ++i)
                l.anchor_mask[i] = j++;
        }
        if (l.w == net_w / 8)
        {
            int j = 0;
            for (int i = 0; i < l.box; ++i)
                l.anchor_mask[i] = j++;
        }
    }
    l.outputs = l.inputs;
    l.output = ( float* )calloc(l.outputs, sizeof(float));

    return l;
}

int entry_index(layer l, int box, int channel, int loc)
{
    return box * l.w * l.h * (4 + l.classes + 1) + channel * l.w * l.h + loc;
}

inline void logistic_cpu(float* input, int size)
{
    for (int i = 0; i < size; ++i)
    {
        input[i] = 1.f / (1.f + expf(-input[i]));
    }
}

inline float logistic_cpu(float input)
{
    return 1.f / (1.f + expf(-input));
}

void decodebox(layer l, box& b, int box_index, int row, int col, int input_w, int input_h)
{
    b.x = (col + logistic_cpu(b.x)) / l.w;
    b.y = (row + logistic_cpu(b.y)) / l.h;
    b.w = exp(b.w) * l.anchors[2 * l.anchor_mask[box_index]] / input_w;
    b.h = exp(b.h) * l.anchors[2 * l.anchor_mask[box_index] + 1] / input_h;
}

void correct_yolo_boxes(std::vector<detection*>& dets, int n, int w, int h, int netw, int neth)
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
        box b = dets[i]->bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;

        dets[i]->bbox = b;
    }
}

std::vector<detection*> forward_darknet_layer_cpu(const float* input, layer l, int img_w, int img_h, int net_w,
                                                  int net_h, const float s_thresh)
{
    std::vector<detection*> dets;
    memcpy(( void* )l.output, ( void* )input, sizeof(float) * l.inputs);

    for (int i = 0; i < l.box; i++)
    {
        int index = entry_index(l, i, 4, 0);
        logistic_cpu(l.output + index, l.w * l.h);
        for (size_t loc = 0; loc < (size_t)l.w * l.h; loc++)
        {
            if (l.output[index + loc] > s_thresh)
            {
                /* row col */
                int row = loc / l.w;
                int col = loc % l.w;

                detection* temp_detection = ( detection* )calloc(1, sizeof(detection));

                /* objectness */
                temp_detection->objectness = l.output[index + loc];

                /* bbox */
                temp_detection->bbox.x = l.output[entry_index(l, i, 0, loc)];
                temp_detection->bbox.y = l.output[entry_index(l, i, 1, loc)];
                temp_detection->bbox.w = l.output[entry_index(l, i, 2, loc)];
                temp_detection->bbox.h = l.output[entry_index(l, i, 3, loc)];
                decodebox(l, temp_detection->bbox, i, row, col, net_w, net_h);

                /* classes_prob */
                temp_detection->prob = ( float* )calloc(l.classes, sizeof(float));
                for (int j = 5; j < l.classes + 5; j++)
                {
                    int grid_index = entry_index(l, i, j, loc);
                    logistic_cpu(l.output + grid_index, 1);
                    temp_detection->prob[j - 5] = l.output[grid_index] > s_thresh ? l.output[grid_index] : 0;
                }

                /* classes_num */
                temp_detection->classes = l.classes;

                dets.push_back(temp_detection);
            }
        }
    }

    if (dets.size() > 0)
    {
        correct_yolo_boxes(dets, dets.size(), img_w, img_h, net_w, net_h);
    }

    return dets;
}

int nms_comparator(const detection* pa, const detection* pb)
{
    float diff = 0;
    if (pb->sort_class >= 0)
    {
        diff = pb->prob[pb->sort_class] - pb->prob[pb->sort_class];
    }
    else
    {
        diff = pb->objectness - pb->objectness;
    }
    if (diff < 0)
        return -1;
    else if (diff > 0)
        return 1;
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

void do_nms_sort(std::vector<detection*>& dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i)
    {
        if (dets[i]->objectness == 0)
        {
            detection* swap = dets[i];
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
            dets[i]->sort_class = k;
        }
        std::sort(dets.begin(), dets.end(), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i]->prob[k] == 0)
                continue;
            box a = dets[i]->bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j]->bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j]->prob[k] = 0;
                }
            }
        }
    }
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

static const char* class_names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                    "train", "truck", "boat", "traffic light", "fire hydrant",
                                    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                    "baseball glove", "skateboard", "surfboard", "tennis racket",
                                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                    "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                    "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                    "scissors", "teddy bear", "hair drier", "toothbrush"};

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] [-s size:608:512] \n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;

    int numBBoxes = 3;
    int total_numAnchors = 9;
    int net_h = 608;
    int net_w = 608;
    int repeat_count = 1;
    int num_thread = 1;

    const int classes = 80;
    const float s_thresh = 0.5;
    const float s_hier_thresh = 0.5;
    const float s_nms = 0.45;    

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:s:")) != -1)
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
            case 's':
                net_w = std::strtoul(optarg, nullptr, 10);
                net_h = net_w;
                fprintf(stderr, "set net input size: %d %d\n", net_h, net_w);
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
    for (int i = 0; i < 1; i++)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", 1, 1,
            total_time, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    image img = imread(image_file);
    int output_node_num = get_graph_output_node_number(graph);

    /* save layer */
    std::vector<layer> layers_params;
    layers_params.clear();

    /* save detection reslult */
    std::vector<detection*> detections;
    detections.clear();

    /* decode layer one by one */
    for (int node = 0; node < output_node_num; ++node)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, node, 0);
        int out_dim[4];
        get_tensor_shape(out_tensor, out_dim, 4);
        layer l_params;
        int out_w = out_dim[3];
        int out_h = out_dim[2];
        l_params = make_darknet_layer(out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes);
        layers_params.push_back(l_params);
        float* out_data = ( float* )get_tensor_buffer(out_tensor);
        std::vector<detection*> l_dets = forward_darknet_layer_cpu(out_data, l_params, img.w, img.h, net_w, net_h, s_thresh);
        if (l_dets.empty())
            continue;
        detections.insert(detections.end(), l_dets.begin(), l_dets.end());
    }

    if (detections.empty())
    {
        fprintf(stderr, "no object detect");
        return 0;
    }

    /* do nms */
    do_nms_sort(detections, detections.size(), classes, s_nms);

    /* print output dectections */
    int i, j;
    for (i = 0; i < detections.size(); ++i)
    {
        int cls = -1;
        float best_class_prob = s_thresh;
        for (j = 0; j < classes; ++j)
        {
            if (detections[i]->prob[j] > best_class_prob)
            {
                if (cls < 0)
                {
                    cls = j;
                    best_class_prob = detections[i]->prob[j];
                }
            }
        }
        if (cls >= 0)
        {
            box b = detections[i]->bbox;
            int left = (b.x - b.w / 2.) * img.w;
            int right = (b.x + b.w / 2.) * img.w;
            int top = (b.y - b.h / 2.) * img.h;
            int bot = (b.y + b.h / 2.) * img.h;
            draw_box(img, left, top, right, bot, 2, 125, 0, 125);
            fprintf(stderr, "%2d: %3.0f%%, [%4d,%4d,%4d,%4d], %s\n", cls, best_class_prob * 100, left, top, right, bot, class_names[cls]);
        }

        if (detections[i]->prob)
            free(detections[i]->prob);
    }

    save_image(img, "yolov4_out");

    /* free resource */
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
        if (l.anchors)
            free(l.anchors);
        if (l.anchor_mask)
            free(l.anchor_mask);
    }

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
