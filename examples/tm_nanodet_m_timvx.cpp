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

    /* nhwc to nchw */
    float* _data = (float*)img.data;
    for (int h = 0; h < lb.h; h++)
    {
        for (int w = 0; w < lb.w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * lb.w * 3 + w * 3 + c;
                int out_index = c * lb.h * lb.w + h * lb.w + w;
                lb.data[out_index] = (_data[in_index] - mean[c]) * norm[c];
            }
        }
    }
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
                    cv::Scalar(0.5 / norm[0] + mean[0], 0.5 / norm[0] + mean[0], 0.5 / norm[2] + mean[2]));
    // Letterbox filling
    cv::copyMakeBorder(img, img_pad, pad.h / 2, pad.h - pad.h / 2, pad.w / 2, pad.w - pad.w / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_pad.convertTo(img_pad, CV_32FC3);
    float* _data = (float*)img_pad.data;
    /* nhwc to nchw */
    for (int h = 0; h < lb.h; h++)
    {
        for (int w = 0; w < lb.w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * lb.w * 3 + w * 3 + c;
                int out_index = c * lb.h * lb.w + h * lb.w + w;
                lb.data[out_index] = (_data[in_index] - mean[c]) * norm[c];
            }
        }
    }

    return lb_scale;
}

static void show_usage()
{
    fprintf(stderr, "[Usage]: [-h]\n");
    fprintf(stderr, "   [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] [-o output_file]\n");
}

void get_input_uint8_data(float* input_fp32, uint8_t* input_data, int size, float input_scale, int zero_point)
{
    for (int i = 0; i < size; i++)
    {
        int udata = (round)(input_fp32[i] / input_scale + zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;

        input_data[i] = udata;
    }
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    const char* output_file = "nanodet_m_timvx_out.jpg";

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

    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt)
    {
        fprintf(stderr, "add_context_device VSI DEVICE failed.\n");
        return -1;
    }
    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
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

    std::vector<uint8_t> input_data(img_size);

    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    get_input_uint8_data(lb.data, input_data.data(), img_size, input_scale, input_zero_point);

    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
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
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count, num_thread,
            total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* nanodet_m postprocess */
    std::vector<Object> proposals, objects;
    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        tensor_t cls_tensor = get_graph_tensor(graph, cls_pred_name[stride_index]);
        tensor_t dis_tensor = get_graph_tensor(graph, dis_pred_name[stride_index]);

        int cls_count = get_tensor_buffer_size(cls_tensor) / sizeof(uint8_t);
        int dis_count = get_tensor_buffer_size(dis_tensor) / sizeof(uint8_t);

        float cls_scale = 0.f;
        float dis_scale = 0.f;
        int cls_zero_point = 0;
        int dis_zero_point = 0;

        get_tensor_quant_param(cls_tensor, &cls_scale, &cls_zero_point, 1);
        get_tensor_quant_param(dis_tensor, &dis_scale, &dis_zero_point, 1);

        const uint8_t* cls_pred_u8 = (const uint8_t*)get_tensor_buffer(cls_tensor);
        const uint8_t* dis_pred_u8 = (const uint8_t*)get_tensor_buffer(dis_tensor);

        std::vector<float> cls_pred(cls_count);
        std::vector<float> dis_pred(dis_count);

        for (int c = 0; c < cls_count; c++)
            cls_pred[c] = ((float)cls_pred_u8[c] - (float)cls_zero_point) * cls_scale;

        for (int c = 0; c < dis_count; c++)
            dis_pred[c] = ((float)dis_pred_u8[c] - (float)dis_zero_point) * dis_scale;

        generate_proposals(cls_pred.data(), dis_pred.data(), 1 << (stride_index + 3), lb, prob_threshold, objects);
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
    for (int i = 0; i < count; i++)
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
