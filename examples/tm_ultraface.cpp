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

#include <vector>
#include <algorithm>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT    1
#define DEFAULT_THREAD_COUNT    1
#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

typedef struct FaceInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

//input image size
const int g_tensor_in_w = 320;
const int g_tensor_in_h = 240;

const float g_score_threshold = 0.7f;
const float g_iou_threshold = 0.3f;
const float g_center_variance = 0.1f;
const float g_size_variance = 0.2f;

static void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, int type = blending_nms)
{
    std::sort(input.begin(), input.end(), [](const FaceInfo& a, const FaceInfo& b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++)
    {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++)
        {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > g_iou_threshold)
            {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type)
        {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < buf.size(); i++)
                {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++)
                {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                fprintf(stderr, "wrong type of nms.");
                exit(-1);
            }
        }
    }
}

static void post_process_ultraface(const char* image_file, float *boxs_data, float *scores_data)
{
    image im = imread(image_file);
    int image_h = im.h;
    int image_w = im.w;

    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f}, {32.0f, 48.0f}, {64.0f, 96.0f}, {128.0f, 192.0f, 256.0f}};
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<std::vector<float>> priors = {};
    std::vector<std::vector<float>> featuremap_size;
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<int> w_h_list = {g_tensor_in_w, g_tensor_in_h};

    for (auto size : w_h_list)
    {
        std::vector<float> fm_item;
        for (float stride : strides)
        {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list)
    {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++)
    {
        float scale_w = g_tensor_in_w / shrinkage_size[0][index];
        float scale_h = g_tensor_in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++)
        {
            for (int i = 0; i < featuremap_size[0][index]; i++)
            {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index])
                {
                    float w = k / g_tensor_in_w;
                    float h = k / g_tensor_in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    /* generate prior anchors finished */
    std::vector<FaceInfo> bbox_collection;
    const int num_anchors = priors.size();

    for (int i = 0; i < num_anchors; i++)
    {
        if (scores_data[i * 2 + 1] > g_score_threshold)
        {
            FaceInfo rects;
            float x_center = boxs_data[i * 4] * g_center_variance * priors[i][2] + priors[i][0];
            float y_center = boxs_data[i * 4 + 1] * g_center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxs_data[i * 4 + 2] * g_size_variance) * priors[i][2];
            float h = exp(boxs_data[i * 4 + 3] * g_size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores_data[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }

    std::vector<FaceInfo> face_list;
    nms(bbox_collection, face_list);

    fprintf(stderr, "detected face num: %ld\n", face_list.size());
    for (int i = 0; i < face_list.size(); i++)
    {
        FaceInfo box = face_list[i];
        draw_box(im, box.x1, box.y1, box.x2, box.y2, 4, 255, 0, 0);
        fprintf(stderr, "BOX %.2f:(%.2f, %.2f),(%.2f, %.2f)\n", box.score, box.x1, box.y1, box.x2, box.y2);
    }

    save_image(im, "tengine_example_out");
    free_image(im);
    fprintf(stderr, "======================================\n");
    fprintf(stderr, "[DETECTED IMAGE SAVED]:\n");
    fprintf(stderr, "======================================\n");
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n\
[example]: tm_ultraface -m version-RFB-320_simplified.tmfile -i 1.jpg\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = NULL;
    char* image_file = NULL;
    float mean[3] = {127.f, 127.f, 127.f};
    float scale[3] = {1.0f / 128, 1.0f / 128, 1.0f / 128};

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
                repeat_count = atoi(optarg);
                break;
            case 't':
                num_thread = atoi(optarg);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == NULL)
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
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", model_file);
    if (graph == NULL)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = g_tensor_in_h * g_tensor_in_w * 3;
    int dims[] = {1, 3, g_tensor_in_h, g_tensor_in_w};    // nchw
    float* input_data = ( float* )malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun graph failed\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data(image_file, input_data, g_tensor_in_h, g_tensor_in_w, mean, scale);
    // input rgb
    image swaprgb_img = {0};
    swaprgb_img.c = 3;
    swaprgb_img.w = g_tensor_in_w;
    swaprgb_img.h = g_tensor_in_h;
    swaprgb_img.data = input_data;
    rgb2bgr_premute(swaprgb_img);

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
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the detection result */
    tensor_t boxs_tensor = get_graph_output_tensor(graph, 0, 0);
    tensor_t scores_tensor = get_graph_output_tensor(graph, 1, 0);

    float* boxs_data = (float* )get_tensor_buffer(boxs_tensor);
    float* scores_data = (float* )get_tensor_buffer(scores_tensor);

    post_process_ultraface(image_file, boxs_data, scores_data);

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}