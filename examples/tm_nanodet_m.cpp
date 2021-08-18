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
 * Author: qinhongjie@imilab.com
 * 
 * original model: https://github.com/RangiLyu/nanodet
 */

/* comment: resize image to model input shape directly */
#define TRY_LETTER_BOX
/* comment: do post softmax within model graph(not recommanded) */
#define TRY_POST_SOFTMAX

/* std c++ includes */
#include <vector>
#include <algorithm>
/* opencv includes */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
/* tengine includes */
#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

// tengine output tensor names
const char* cls_pred_name[] = {
    "cls_pred_stride_8", "cls_pred_stride_16", "cls_pred_stride_32"};
const char* dis_pred_name[] = {
#ifdef TRY_POST_SOFTMAX
    "dis_pred_stride_8", "dis_pred_stride_16", "dis_pred_stride_32"
#else  /* !TRY_POST_SOFTMAX */
    "dis_sm_stride_8", "dis_sm_stride_16", "dis_sm_stride_32"
#endif /* TRY_POST_SOFTMAX */
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static __inline float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template<typename _Tp>
static int softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp max_value = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i)
    {
        dst[i] = std::exp /*fast_exp*/ (src[i] - max_value);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// @brief:  generate and filter proposals
// @param:  cls_pred[in] (1, num_grid, cls_num)
// @param:  dis_pred[in] (1, num_grid, 4*reg_max)
// @param:  stride[in]   P3/8, P4/16, P5/32
// @param:  in_pad[in]   as letter box's shape
// @param:  prob_threshold[in]
// @param:  objects[out] output detected objects
static void generate_proposals(const float* cls_pred, const float* dis_pred, int stride,
                               const image& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid_x = in_pad.w / stride;
    const int num_grid_y = in_pad.h / stride;
    // Note: Here, we hard coded some model parameters for simplicity.
    // Call api "get_tensor_shape" to get more model information if necessary.
    const int num_class = 80; // coco dataset
    // Discrete distribution parameter, see the following resources for more details:
    // [nanodet-m.yml](https://github.com/RangiLyu/nanodet/blob/main/config/nanodet-m.yml)
    // [GFL](https://arxiv.org/pdf/2006.04388.pdf)
    const int reg_max_1 = 8; // 32 / 4;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const int idx = i * num_grid_x + j;

            const float* scores = cls_pred + idx * num_class;

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                if (scores[k] > score)
                {
                    label = k;
                    score = scores[k];
                }
            }

            if (score >= prob_threshold)
            {
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    // predicted distance distribution after softmax
#ifdef TRY_POST_SOFTMAX
                    float dis_after_sm[8] = {0.};
                    softmax(dis_pred + idx * reg_max_1 * 4 + k * reg_max_1, dis_after_sm, 8);
#else  /* !TRY_POST_SOFTMAX */
                    const float* dis_after_sm = dis_pred + idx * reg_max_1 * 4 + k * reg_max_1;
#endif /* TRY_POST_SOFTMAX */
                    // integral on predicted discrete distribution
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                        //printf("%2.6f ", dis_after_sm[l]);
                    }
                    //printf("\n");

                    pred_ltrb[k] = dis * stride;
                }

                // predict box center point
                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0]; // left
                float y0 = pb_cy - pred_ltrb[1]; // top
                float x1 = pb_cx + pred_ltrb[2]; // right
                float y1 = pb_cy + pred_ltrb[3]; // bottom

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, const char* path)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.3f%%, [%7.3f, %7.3f, %7.3f, %7.3f], %s\n",
                obj.label, obj.prob * 100, obj.rect.x, obj.rect.y,
                obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

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

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    //cv::imshow("image", image);
    //cv::waitKey(0);
    cv::imwrite(path, image);
}

/// @brief change layout, from nhwc to nchw
/// @param src original tensor buffer
/// @param dst result tensor buffer (should not be same as src)
/// @param h_limit height limitation
/// @param w_limit width limitation
/// @param c_limit channel limitation
/// @param mean mean values per channel
/// @param norm norm values per channel
static void nhwc_to_nchw(float* src, float* dst, int h_limit, int w_limit, int c_limit, const float* mean, const float* norm)
{
    for (int h = 0; h < h_limit; h++)
    {
        for (int w = 0; w < w_limit; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * w_limit * 3 + w * 3 + c;
                int out_index = c * h_limit * w_limit + h * w_limit + w;
                dst[out_index] = (src[in_index] - mean[c]) * norm[c];
            }
        }
    }
}

// @brief:  get input data and resize to model input shape directly
static int get_input_data(const char* path, const float* mean, const float* norm, image& lb)
{
    // load input image
    cv::Mat img = cv::imread(path, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", path);
        return -1;
    }

    if (img.cols != lb.w || img.rows != lb.h) cv::resize(img, img, cv::Size(lb.w, lb.h));
    img.convertTo(img, CV_32FC3);
    float* _data = (float*)img.data;

    nhwc_to_nchw(_data, lb.data, lb.h, lb.w, 3, mean, norm);
    return 0;
}

// @brief:  get input data, keep aspect ratio and fill to the center of letter box
// @param:  lb[in/out]  letter box image inst
// @param:  pad[out]    top and left pad size
// @return: resize scale from origin image to letter box
static float get_input_data(const char* path, const float* mean, const float* norm, image& lb, image& pad)
{
    // load input image
    cv::Mat img = cv::imread(path, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", path);
        return -1.;
    }

    /* letterbox process to support different letterbox size */
    float lb_scale_w = lb.w * 1. / img.cols, lb_scale_h = lb.h * 1. / img.rows;
    float lb_scale = lb_scale_w < lb_scale_h ? lb_scale_w : lb_scale_h;
    int w = lb_scale * img.cols;
    int h = lb_scale * img.rows;

    if (w != lb.w || h != lb.h) cv::resize(img, img, cv::Size(w, h));
    img.convertTo(img, CV_32FC3);

    // Pad to letter box rectangle
    pad.w = lb.w - w; //(w + 31) / 32 * 32 - w;
    pad.h = lb.h - h; //(h + 31) / 32 * 32 - h;
    // Generate a gray image using opencv
    cv::Mat img_pad(lb.w, lb.h, CV_32FC3, //cv::Scalar(0));
                    cv::Scalar(0.5 / norm[0] + mean[0], 0.5 / norm[1] + mean[1], 0.5 / norm[2] + mean[2]));
    // Letterbox filling
    cv::copyMakeBorder(img, img_pad, pad.h / 2, pad.h - pad.h / 2, pad.w / 2, pad.w - pad.w / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_pad.convertTo(img_pad, CV_32FC3);
    float* _data = (float*)img_pad.data;
    nhwc_to_nchw(_data, lb.data, lb.h, lb.w, 3, mean, norm);

    return lb_scale;
}

static void show_usage()
{
    fprintf(stderr, "[Usage]: [-h]\n");
    fprintf(stderr, "   [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] [-o output_file]\n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* output_file = "nanodet_m_out.jpg";

    const float mean[3] = {103.53f, 116.28f, 123.675f}; // bgr
    const float norm[3] = {0.017429f, 0.017507f, 0.017125f};

    int repeat_count = 1;
    int num_thread = 1;

    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    int res;
    while ((res = getopt(argc, argv, "m:i:o:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'o':
            output_file = optarg;
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
    if (nullptr == model_file || nullptr == image_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }
    if (!check_file_exist(model_file) || !check_file_exist(image_file))
    {
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (0 != init_tengine())
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (nullptr == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* get input tensor of graph */
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    /* get shape of input tensor */
    int i, dims[4]; // nchw
    int dim_num = get_tensor_shape(input_tensor, dims, 4);
    if (4 != dim_num)
    {
        fprintf(stderr, "Get input tensor shape error\n");
        return -1;
    }

    image lb = make_image(dims[3], dims[2], dims[1]);
    int img_size = lb.w * lb.h * lb.c;

#ifdef TRY_LETTER_BOX
    image pad = make_empty_image(lb.w, lb.h, lb.c);
    float lb_scale = get_input_data(image_file, mean, norm, lb, pad);
#else  /* !TRY_LETTER_BOX */
    get_input_data(image_file, mean, norm, lb);
#endif /* TRY_LETTER_BOX */

    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, lb.data, img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph to infer shape, and set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (i = 0; i < repeat_count; i++)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* nanodet_m postprocess */
    std::vector<Object> proposals, objects;
    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        tensor_t cls_tensor = get_graph_tensor(graph, cls_pred_name[stride_index]);
        tensor_t dis_tensor = get_graph_tensor(graph, dis_pred_name[stride_index]);
        if (NULL == cls_tensor || NULL == dis_tensor)
        {
            fprintf(stderr, "get graph tensor failed\n");
            return -1;
        }
        const float* cls_pred = (const float*)get_tensor_buffer(cls_tensor);
        const float* dis_pred = (const float*)get_tensor_buffer(dis_tensor);
        generate_proposals(cls_pred, dis_pred, 1 << (stride_index + 3),
                           lb, prob_threshold, objects);
        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    cv::Mat img = cv::imread(image_file);
    int count = picked.size();
    fprintf(stderr, "detection num: %d\n", count);

    objects.resize(count);
    for (i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

#ifdef TRY_LETTER_BOX
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (pad.w / 2)) / lb_scale;
        float y0 = (objects[i].rect.y - (pad.h / 2)) / lb_scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (pad.w / 2)) / lb_scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (pad.h / 2)) / lb_scale;
#else  /* !TRY_LETTER_BOX */
        // adjust offset to original unresized
        static float lb_scale_w = 1. * lb.w / img.cols;
        static float lb_scale_h = 1. * lb.h / img.rows;
        float x0 = (objects[i].rect.x) / lb_scale_w;
        float y0 = (objects[i].rect.y) / lb_scale_h;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / lb_scale_w;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / lb_scale_h;
#endif /* TRY_LETTER_BOX */

        // clip
        x0 = std::max(std::min(x0, (float)(img.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img.rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    draw_objects(img, objects, output_file);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
    return 0;
}
