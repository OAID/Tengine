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
 * Author: guanguojing1989@126.com
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"


static constexpr int kAnchorNum = 3;
static constexpr int kClassNum = 80;
static constexpr float kIgnoreThresh = 0.5f;
static constexpr float kClassThresh = 0.5f;
static constexpr float kNms = 0.45f;

struct YoloKernel
{
    int scale;
    float anchors[kAnchorNum * 2];
};

static constexpr YoloKernel yolo1 = {32, {116, 90, 156, 198, 373, 326}};
static constexpr YoloKernel yolo2 = {16, {30, 61, 62, 45, 59, 119}};
static constexpr YoloKernel yolo3 = {8, {10, 13, 16, 30, 33, 23}};

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    int classes;
    float prob;
    float objectness;
} detection;

inline float logistic_cpu(const float input)
{
    return 1.f / (1.f + expf(-input));
}

void correct_yolo_boxes(std::vector<detection>& dets, int w, int h, int netw, int neth)
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

    for (i = 0; i < dets.size(); ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2.) / (( float )new_w / w);
        b.y = (b.y - (neth - new_h) / 2.) / (( float )new_h / h);
        b.w /= (( float )new_w / w);
        b.h /= (( float )new_h / h);

        dets[i].bbox = b;
    }
}

std::vector<detection> forward_darknet_layer_cpu(const float* input, int img_w, int img_h, int net_w, int net_h, int out_w, int out_h)
{
    std::vector<detection> detections;
    const YoloKernel* yolo_kernel;
    const int kernel_scale = net_w / out_w;
    if (kernel_scale == yolo1.scale)
    {
        yolo_kernel = &yolo1;
    }
    else if (kernel_scale == yolo2.scale)
    {
        yolo_kernel = &yolo2;
    }
    else if (kernel_scale == yolo3.scale)
    {
        yolo_kernel = &yolo3;
    }
    else
    {
        fprintf(stderr, "Kernel parameter is not found!");
        return detections;
    }

    for (int shift_y = 0; shift_y < out_h; shift_y++)
    {
        for (int shift_x = 0; shift_x < out_w; shift_x++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                const float* pdata = input + channel * out_h * out_w * (kClassNum + 5) +
                                      shift_y * out_w * (kClassNum + 5) + shift_x * (kClassNum + 5);
                float box_prob = logistic_cpu(*(pdata + 4));
                if (box_prob < kIgnoreThresh)
                    continue;

                int class_id = 0;
                float max_class_prob = 0.f;
                for (int cls = 5; cls < (kClassNum + 5); cls++)
                {
                    float class_prob = logistic_cpu(*(pdata + cls));
                    if (class_prob > max_class_prob)
                    {
                        max_class_prob = class_prob;
                        class_id = cls - 5;
                    }
                }

                if (max_class_prob < kClassThresh)
                    continue;

                detection det;
                det.classes = class_id;
                det.prob = max_class_prob;
                det.objectness = box_prob;
                float center_x, center_y, box_w, box_h;

                center_x = (shift_x - 0.5f + 2.0f * logistic_cpu(*(pdata + 0))) * yolo_kernel->scale;
                center_y = (shift_y - 0.5f + 2.0f * logistic_cpu(*(pdata + 1))) * yolo_kernel->scale;

                box_w = 2.0f * logistic_cpu(*(pdata + 2));
                box_w = box_w * box_w * yolo_kernel->anchors[2 * channel];

                box_h = 2.0f * logistic_cpu(*(pdata + 3));
                box_h = box_h * box_h * yolo_kernel->anchors[2 * channel + 1];

                det.bbox.x = center_x - box_w / 2;
                det.bbox.y = center_y - box_h / 2;
                det.bbox.w = box_w;
                det.bbox.h = box_h;
                detections.push_back(std::move(det));
            }
        }
    }

    correct_yolo_boxes(detections, img_w, img_h, net_w, net_h);
    return std::move(detections);
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

std::vector<detection> do_nms_sort(std::vector<detection>& dets, int classes, float thresh)
{
    std::vector<detection> nms_detections;

    for (int k = 0; k < classes; ++k)
    {
        std::vector<detection> class_detection;
        for (auto & det : dets)
        {
            if (det.classes == k)
            {
                class_detection.push_back(det);
            }
        }

        std::sort(class_detection.begin(), class_detection.end(), [](const detection & a, const detection & b) {
            return a.prob > b.prob;
        });

        std::vector<detection> nms_detection;
        for (int i = 0; i < class_detection.size(); i++)
        {
            if (class_detection[i].prob < 0) continue;
            nms_detection.push_back(class_detection[i]);
            box a = class_detection[i].bbox;
            for (int j = i + 1; j < class_detection.size(); j++)
            {
                box b = class_detection[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    class_detection[j].prob = -1;
                }
            }
        }
        nms_detections.insert(nms_detections.end(), nms_detection.begin(), nms_detection.end());
    }

    return std::move(nms_detections);
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
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] [-s size:640:640] \n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    int net_h = 640;
    int net_w = 640;
    int repeat_count = 1;
    int num_thread = 1;

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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
            repeat_count, num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    image img = imread(image_file);
    int output_node_num = get_graph_output_node_number(graph);

    /* save detection reslult */
    std::vector<detection> detections;

    /* decode layer one by one*/
    for (int node = 0; node < output_node_num; ++node)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, node, 0);
        int out_dim[5];
        get_tensor_shape(out_tensor, out_dim, 5);

        float* out_data = ( float* )get_tensor_buffer(out_tensor);
        int out_w = out_dim[3];
        int out_h = out_dim[2];
        auto node_detection = forward_darknet_layer_cpu(out_data, img.w, img.h, net_w, net_h, out_w, out_h);
        detections.insert(detections.end(), node_detection.begin(), node_detection.end());
    }

    if (detections.size() == 0)
    {
        fprintf(stderr, "no object detect");
        return 0;
    }

    /* do nms */
    auto nms_detections = do_nms_sort(detections, kClassNum, kNms);

    for (auto& det : nms_detections)
    {
        int left = det.bbox.x;
        int right = det.bbox.x + det.bbox.w;
        int top = det.bbox.y;
        int bot = det.bbox.y + det.bbox.h;
        draw_box(img, left, top, right, bot, 2, 125, 0, 125);
        fprintf(stderr, "left = %d,right = %d,top = %d,bot = %d, prob = %f class id = %d\n",
                left, right, top, bot,
                det.prob, det.classes);
    }

    save_image(img, "tengine_example_out");

    /* free resource */
    /* release tengine */
    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
        release_graph_tensor(out_tensor);
    }

    free_image(img);

    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
