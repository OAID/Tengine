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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: john2357@163.com
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

enum
{
    YOLOV3 = 0,
    YOLO_FASTEST = 1,
    YOLO_FASTEST_XL = 2
};

using namespace std;

struct BBoxRect
{
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float area;
    int label;
};

struct TMat
{
    operator const float*() const
    {
        return (const float*)data;
    }

    float *row(int row) const
    {
        return (float *)data + w * row;
    }

    TMat channel_range(int start, int chn_num) const 
    {
        TMat mat = { 0 };

        mat.batch = 1;
        mat.c = chn_num;
        mat.h = h;
        mat.w = w;
        mat.data = (float *)data + start * h * w;

        return mat;
    }

    TMat channel(int channel) const
    {
        return channel_range(channel, 1);
    }

    int batch, c, h, w;
    void *data;
};

class Yolov3DetectionOutput
{
public:
    int init(int version);
    int forward(const std::vector<TMat>& bottom_blobs, std::vector<TMat>& top_blobs);
private:

    int m_num_box;
    int m_num_class;
    float m_anchors_scale[32];
    float m_biases[32];
    int m_mask[32];
    float m_confidence_threshold;
    float m_nms_threshold;
};

int Yolov3DetectionOutput::init(int version)
{
    memset(this, 0, sizeof(*this));
    m_num_box = 3;
    m_num_class = 80;
	
	fprintf(stderr, "Yolov3DetectionOutput init param[%d]\n", version);
	
    if (version == YOLOV3)
    {
        m_anchors_scale[0] = 32;
        m_anchors_scale[1] = 16;
        m_anchors_scale[2] = 8;

        float bias[] = { 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 };
        memcpy(m_biases, bias, sizeof(bias));

        m_mask[0] = 6;
        m_mask[1] = 7;
        m_mask[2] = 8;

        m_mask[3] = 3;
        m_mask[4] = 4;
        m_mask[5] = 5;

        m_mask[6] = 0;
        m_mask[7] = 1;
        m_mask[8] = 2;
    }
    else if (version == YOLO_FASTEST || version == YOLO_FASTEST_XL)
    {
        m_anchors_scale[0] = 32;
        m_anchors_scale[1] = 16;

        float bias[] = { 12, 18,  37, 49,  52,132, 115, 73, 119,199, 242,238 };
        memcpy(m_biases, bias, sizeof(bias));

        m_mask[0] = 3;
        m_mask[1] = 4;
        m_mask[2] = 5;

        m_mask[3] = 0;
        m_mask[4] = 1;
        m_mask[5] = 2;
    }

    m_confidence_threshold = 0.48f;
    m_nms_threshold = 0.45f;

    return 0;
}

static inline float intersection_area(const BBoxRect& a, const BBoxRect& b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<BBoxRect>& datas, int left, int right)
{
    int i = left;
    int j = right;
    float p = datas[(left + right) / 2].score;

    while (i <= j)
    {
        while (datas[i].score > p)
            i++;

        while (datas[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, left, j);

    if (i < right)
        qsort_descent_inplace(datas, i, right);
}

static void qsort_descent_inplace(std::vector<BBoxRect>& datas)
{
    if (datas.empty())
        return;

    qsort_descent_inplace(datas, 0, (int)(datas.size() - 1));
}

static void nms_sorted_bboxes(std::vector<BBoxRect>& bboxes, std::vector<size_t>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = bboxes.size();

    for (size_t i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area > nms_threshold * union_area)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return (float)(1.f / (1.f + exp(-x)));
}

int Yolov3DetectionOutput::forward(const std::vector<TMat>& bottom_blobs, std::vector<TMat>& top_blobs)
{
    // gather all box
    std::vector<BBoxRect> all_bbox_rects;

    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        std::vector<std::vector<BBoxRect> > all_box_bbox_rects;
        all_box_bbox_rects.resize(m_num_box);
        const TMat& bottom_top_blobs = bottom_blobs[b];

        int w = bottom_top_blobs.w;
        int h = bottom_top_blobs.h;
        int channels = bottom_top_blobs.c;
        //printf("%d %d %d\n", w, h, channels);
        const int channels_per_box = channels / m_num_box;

        // anchor coord + box score + num_class
        if (channels_per_box != 4 + 1 + m_num_class)
            return -1;
        size_t mask_offset = b * m_num_box;
        int net_w = (int)(m_anchors_scale[b] * w);
        int net_h = (int)(m_anchors_scale[b] * h);
        //printf("%d %d\n", net_w, net_h);

        //printf("%d %d %d\n", w, h, channels);
        for (int pp = 0; pp < m_num_box; pp++)
        {
            int p = pp * channels_per_box;
            int biases_index = (int)(m_mask[pp + mask_offset]);
            //printf("%d\n", biases_index);
            const float bias_w = m_biases[biases_index * 2];
            const float bias_h = m_biases[biases_index * 2 + 1];
            //printf("%f %f\n", bias_w, bias_h);
            const float* xptr = bottom_top_blobs.channel(p);
            const float* yptr = bottom_top_blobs.channel(p + 1);
            const float* wptr = bottom_top_blobs.channel(p + 2);
            const float* hptr = bottom_top_blobs.channel(p + 3);

            const float* box_score_ptr = bottom_top_blobs.channel(p + 4);

            // softmax class scores
            TMat scores = bottom_top_blobs.channel_range(p + 5, m_num_class);
            //softmax->forward_inplace(scores, opt);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int q = 0; q < m_num_class; q++)
                    {
                        float score = scores.channel(q).row(i)[j];
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                    }

                    //sigmoid(box_score) * sigmoid(class_score)
                    float confidence = 1.f / ((1.f + exp(-box_score_ptr[0]) * (1.f + exp(-class_score))));
                    if (confidence >= m_confidence_threshold)
                    {
                        // region box
                        float bbox_cx = (j + sigmoid(xptr[0])) / w;
                        float bbox_cy = (i + sigmoid(yptr[0])) / h;
                        float bbox_w = (float)(exp(wptr[0]) * bias_w / net_w);
                        float bbox_h = (float)(exp(hptr[0]) * bias_h / net_h);

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;

                        float area = bbox_w * bbox_h;

                        BBoxRect c = { confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area, class_index };
                        all_box_bbox_rects[pp].push_back(c);
                    }

                    xptr++;
                    yptr++;
                    wptr++;
                    hptr++;

                    box_score_ptr++;
                }
            }
        }

        for (int i = 0; i < m_num_box; i++)
        {
            const std::vector<BBoxRect>& box_bbox_rects = all_box_bbox_rects[i];

            all_bbox_rects.insert(all_bbox_rects.end(), box_bbox_rects.begin(), box_bbox_rects.end());
        }
    }

    // global sort inplace
    qsort_descent_inplace(all_bbox_rects);

    // apply nms
    std::vector<size_t> picked;
    nms_sorted_bboxes(all_bbox_rects, picked, m_nms_threshold);

    // select
    std::vector<BBoxRect> bbox_rects;

    for (size_t i = 0; i < picked.size(); i++)
    {
        size_t z = picked[i];
        bbox_rects.push_back(all_bbox_rects[z]);
    }

    // fill result
    int num_detected = (int)(bbox_rects.size());
    if (num_detected == 0)
        return 0;

    TMat& top_blob = top_blobs[0];

    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        float score = r.score;
        float* outptr = top_blob.row(i);

        outptr[0] = (float)(r.label + 1); // +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }
    top_blob.h = num_detected;

    return 0;
}

static void get_input_data_darknet(const char* image_file, float* input_data, int net_h, int net_w)
{
    float mean[3] = { 0.f, 0.f, 0.f };
    float scale[3] = { 1.0f / 255, 1.0f / 255, 1.0f / 255 };

    //no letter box by default
    get_input_data(image_file, input_data, net_h, net_w, mean, scale);
    // input rgb
    image swaprgb_img = { 0 };
    swaprgb_img.c = 3;
    swaprgb_img.w = net_w;
    swaprgb_img.h = net_h;
    swaprgb_img.data = input_data;
    rgb2bgr_premute(swaprgb_img);
}

static void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

static void run_yolo(graph_t graph, std::vector<BBoxRect> &boxes, int img_width, int img_height)
{
    Yolov3DetectionOutput yolo;
    std::vector<TMat> yolo_inputs, yolo_outputs;
    
    yolo.init(YOLO_FASTEST);

    int output_node_num = get_graph_output_node_number(graph);
    yolo_inputs.resize(output_node_num);
    yolo_outputs.resize(1);

    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
        int out_dim[4] = { 0 };
        get_tensor_shape(out_tensor, out_dim, 4);

        yolo_inputs[i].batch = out_dim[0];
        yolo_inputs[i].c = out_dim[1];
        yolo_inputs[i].h = out_dim[2];
        yolo_inputs[i].w = out_dim[3];
        yolo_inputs[i].data = get_tensor_buffer(out_tensor);
    }

    std::vector<float> output_buf;

    output_buf.resize(1000 * 6, 0);
    yolo_outputs[0].batch = 1;
    yolo_outputs[0].c = 1;
    yolo_outputs[0].h = 1000;
    yolo_outputs[0].w = 6;
    yolo_outputs[0].data = output_buf.data();

    yolo.forward(yolo_inputs, yolo_outputs);

    //image roi on net input
    bool letterbox = false;
    float roi_left = 0.f, roi_top = 0.f, roi_width = 1.f, roi_height = 1.f;

    if (letterbox)
    {
        if (img_width > img_height)
        {
            roi_height = img_height / (float)img_width;
            roi_top = (1 - roi_height) / 2;
        }
        else
        {
            roi_width = img_width / (float)img_height;
            roi_left = (1 - roi_width) / 2;
        }
    }

    //rect correct
    for (int i = 0; i < yolo_outputs[0].h; i++)
    {
        float *data_row = yolo_outputs[0].row(i);

        BBoxRect box = { 0 };
        box.score = data_row[1];
        box.label = data_row[0];
        box.xmin = (data_row[2] - roi_left) / roi_width * img_width;
        box.ymin = (data_row[3] - roi_top) / roi_height * img_height;
        box.xmax = (data_row[4] - roi_left) / roi_width * img_width;
        box.ymax = (data_row[5] - roi_top) / roi_height * img_height;

        boxes.push_back(box);
    }

    //release
    for (int i = 0; i < output_node_num; ++i)
    {
        tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
        release_graph_tensor(out_tensor);
    }
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;

    int net_w = 320;
    int net_h = 320;

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
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = net_h * net_w * 3;
    int dims[] = { 1, 3, net_h, net_w };    // nchw

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
    std::vector<BBoxRect> boxes;
    run_yolo(graph, boxes, img.w, img.h);

    for (int i = 0; i < (int)boxes.size(); ++i)
    {
        BBoxRect b = boxes[i];
        draw_box(img, b.xmin, b.ymin, b.xmax, b.ymax, 2, 125, 0, 125);
        fprintf(stderr, "class=%2d score=%.2f left = %.2f,right = %.2f,top = %.2f,bot = %.2f\n", b.label, b.score, b.xmin, b.xmax, b.ymin, b.ymax);
    }
    save_image(img, "tengine_example_out");

    /* release tengine */
    free_image(img);

    release_graph_tensor(input_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
