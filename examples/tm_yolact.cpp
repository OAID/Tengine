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
 * Author: qtang@openailab.com
 */

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/examples/yolact.cpp
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */


#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tengine_c_api.h"


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

void get_input_data_cv(const cv::Mat& sample, float* input_data, int img_h, int img_w, const float* mean, const float* scale, int swapRB = 0)
{
    cv::Mat img;
    if(sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
    }
    else if(sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
    }
    else if(sample.channels() == 3 && swapRB == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else
    {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = (*img_data - mean[c]) * scale[c];
                img_data++;
            }
        }
    }
}

struct Box2f
{
    float cx;
    float cy;
    float w;
    float h;
};

static std::vector<Box2f> generate_priorbox(int num_priores)
{
    std::vector<Box2f> priorboxs(num_priores);

    const int conv_ws[5] = {69, 35, 18, 9, 5};
    const int conv_hs[5] = {69, 35, 18, 9, 5};

    const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
    const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

    int index = 0;

    for(int i = 0; i < 5; i++)
    {
        int conv_w = conv_ws[i];
        int conv_h = conv_hs[i];
        int scale = scales[i];
        for(int ii = 0; ii < conv_h; ii++)
        {
            for(int j = 0; j < conv_w; j++)
            {
                float cx = (j + 0.5f) / conv_w;
                float cy = (ii + 0.5f) / conv_h;

                for (int k = 0; k < 3; k++)
                {
                    float ar = aspect_ratios[k];

                    ar = sqrt(ar);

                    float w = scale * ar / 550;
                    float h = scale / ar / 550;

                    h = w;

                    Box2f& priorbox = priorboxs[index];       

                    priorbox.cx = cx;
                    priorbox.cy = cy;
                    priorbox.w = w;
                    priorbox.h = h;

                    index += 1;
                }
            }
        }
    }

    return priorboxs;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


static void fast_nms(std::vector< std::vector<Object> >& class_candidates, std::vector<Object>& objects, const float iou_thresh, const int nms_top_k, const int keep_top_k)
{
    for(int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidate = class_candidates[i];
        std::sort(candidate.begin(), candidate.end(),
                  [](const Object& a, const Object& b) {return a.prob > b.prob;});
        if (candidate.size() == 0)
            continue;

        if(nms_top_k != 0&& nms_top_k > candidate.size())
        {
            candidate.erase(candidate.begin()+nms_top_k, candidate.end());
        }

        objects.push_back(candidate[0]);
        const int n = candidate.size();
        std::vector<float> areas(n);
        std::vector<int> keep(n);
        for(int j = 0; j < n; j++)
        {
            areas[j] = candidate[j].rect.area();
        }
        std::vector< std::vector<float> > iou_matrix;
        for(int j = 0; j < n; j++)
        {
            std::vector<float> iou_row(n);
            for(int k = 0; k < n; k++)
            {
                float inter_area = intersection_area(candidate[j], candidate[k]);
                float union_area = areas[j] + areas[k] - inter_area;
                iou_row[k] = inter_area / union_area;
            }
            iou_matrix.push_back(iou_row);
        }
        for(int j = 1; j < n; j++)
        {
            std::vector<float>::iterator max_value;
            max_value = std::max_element(iou_matrix[j].begin(), iou_matrix[j].begin()+j-1);
            if(*max_value <= iou_thresh)
            {
                objects.push_back(candidate[j]);
            }
        }

    }
    std::sort(objects.begin(), objects.end(),
              [](const Object& a, const Object& b) {return a.prob > b.prob;});
    if(objects.size() > keep_top_k)
        objects.resize(keep_top_k);
}

static int detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects)
{
    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    const int target_size = 550;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0/58.40f, 1.0/57.12f, 1.0/57.38f};

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(NULL, "tengine", "./yolact_tm.tmfile");
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size      = target_size * target_size * 3;
    int dims[]        = {1, 3, target_size, target_size};    // nchw
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

    if (prerun_graph_multithread(graph, TENGINE_CLUSTER_ALL, 1) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_cv(bgr, input_data, target_size, target_size, mean_vals, norm_vals, 1);
    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    dump_graph(graph);

    /* run graph */
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
        return -1;
    }

    /* get the result of classification */
    tensor_t maskmaps_tensor   = get_graph_output_tensor(graph, 0, 0);
    tensor_t location_tensor   = get_graph_output_tensor(graph, 1, 0);
    tensor_t mask_tensor       = get_graph_output_tensor(graph, 2, 0);
    tensor_t confidence_tensor = get_graph_output_tensor(graph, 3, 0);
    float* maskmaps     = ( float* )get_tensor_buffer(maskmaps_tensor);
    float* location     = ( float* )get_tensor_buffer(location_tensor);
    float* mask         = ( float* )get_tensor_buffer(mask_tensor);
    float* confidence   = ( float* )get_tensor_buffer(confidence_tensor);

    int num_class = 81;
    int num_priors = 19248;
    std::vector<Box2f> priorboxes = generate_priorbox(num_priors);
    const float confidence_thresh = 0.05f;
    const float nms_thresh = 0.5f;
    const int keep_top_k = 200;

    std::vector< std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);

    for(int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence + i * 81;
        const float* loc  = location + i * 4;
        const float* maskdata = mask + i * 32;
        Box2f& priorbox = priorboxes[i];

        int label = 0;
        float score = 0.f;
        for(int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if(class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        if(label == 0||score <= confidence_thresh)
            continue;

        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float bbox_cx = var[0] * loc[0] * priorbox.w + priorbox.cx;
        float bbox_cy = var[1] * loc[1] * priorbox.h + priorbox.cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * priorbox.w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * priorbox.h);  

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, (float)(bgr.rows - 1)), 0.f);

        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2-obj_x1+1, obj_y2-obj_y1+1);
        obj.label = label;
        obj.prob = score;

        obj.maskdata = std::vector<float>(maskdata, maskdata + 32);

        class_candidates[label].push_back(obj);
    }

    objects.clear();
    fast_nms(class_candidates, objects, nms_thresh, 0, keep_top_k);


    for (int i=0; i<objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask1(138, 138, CV_32FC1);
        {
            mask1 = cv::Scalar(0.f);

            for (int p=0; p<32; p++)
            {
                const float* maskmap = maskmaps + p;
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask1.data;

                // mask += m * coeff
                for (int j=0; j< 138 * 138; j++)
                {
                    mp[j] += maskmap[j*32] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask1, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y=0; y<img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x=0; x<img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    /* release tengine */
    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
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

    static const unsigned char colors[19][3] = {
        {244,  67,  54},
        {233,  30,  99},
        {156,  39, 176},
        {103,  58, 183},
        { 63,  81, 181},
        { 33, 150, 243},
        {  3, 169, 244},
        {  0, 188, 212},
        {  0, 150, 136},
        { 76, 175,  80},
        {139, 195,  74},
        {205, 220,  57},
        {255, 235,  59},
        {255, 193,   7},
        {255, 152,   0},
        {255,  87,  34},
        {121,  85,  72},
        {158, 158, 158},
        { 96, 125, 139}
    };

    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index++];

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // draw mask
        for (int y=0; y<image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x=0; x<image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::imwrite("result.png", image);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolact(m, objects);
    draw_objects(m, objects);

    return 0;
}
