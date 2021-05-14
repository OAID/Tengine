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
 * Author: guanguojing1989@126.com
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_IMG_H 320
#define DEFAULT_IMG_W 256
#define DEFAULT_SCALE1 (0.0039216)
#define DEFAULT_SCALE2 (0.0039215)
#define DEFAULT_SCALE3 (0.0039215)
#define DEFAULT_MEAN1 0.406
#define DEFAULT_MEAN2 0.457
#define DEFAULT_MEAN3 0.480
#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

using bbox_t = std::array<float, 4>;
using pt_t = std::array<float, 2>;
using predict_t = std::tuple<cv::Mat, cv::Mat, cv::Mat>;

const float s_keypoint_thresh = 0.2;

cv::Mat get_3rd_point(const cv::Mat & a, const cv::Mat & b)
{
    auto direct = a - b;
    cv::Mat result(direct.size(), direct.type());
    result.row(0).col(0) = b.row(0).col(0) - direct.row(0).col(1);
    result.row(0).col(1) = b.row(0).col(1) + direct.row(0).col(0);
    return result;
}

cv::Mat get_input_data_pose(const char * img_file_path)
{
    cv::Mat img = cv::imread(img_file_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);

    float* img_data = ( float* )img.data;
    float means[3]{DEFAULT_MEAN1, DEFAULT_MEAN2, DEFAULT_MEAN3};
    float scales[3]{DEFAULT_SCALE1, DEFAULT_SCALE2, DEFAULT_SCALE3};

    for (int h = 0; h < img.rows; h++)
    {
        for (int w = 0; w < img.cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                *img_data = (scales[c] * (*img_data)) - means[c];
                img_data++;
            }
        }
    }

    return std::move(img);
}

cv::Mat crop_box(const cv::Mat & org_img,
                           const pt_t & up_left,
                           const pt_t & bottom_right,
                           const int & input_res_h,
                           const int & input_res_w)
{
    auto img = org_img.clone();

    std::array<int, 2> ul{static_cast<int>(up_left[0]), static_cast<int>(up_left[1])};
    std::array<int, 2> br{static_cast<int>(bottom_right[0] - 1), static_cast<int>(bottom_right[1] - 1)};

    auto len_h = std::max((br[1] - ul[1]) * 1.f, (br[0] - ul[0]) * input_res_h * 1.f / input_res_w);
    auto len_w = len_h * input_res_w / input_res_h;

    std::vector<size_t> box_shape = {static_cast<size_t>((br[1] - ul[1])), static_cast<size_t>((br[0] - ul[0]))};
    std::vector<int> pad_size = {static_cast<int>((len_h - box_shape[0]) / 2),
                                 static_cast<int>((len_w - box_shape[1]) / 2)};
    // padding zero
    if (ul[1] > 0)
    {
        img.rowRange(0, ul[1]) = 0.f;
    }

    if (ul[0] > 0)
    {
        img.colRange(0, ul[0]) = 0.f;
    }

    if (br[1] < (img.rows - 1))
    {
        img.rowRange(br[1] + 1, img.rows - 1) = 0.f;
    }

    if (br[0] < (img.cols - 1))
    {
        img.colRange(br[0] + 1, img.cols - 1) = 0.f;
    }

    cv::Mat src = cv::Mat::zeros(3, 2, CV_32FC1);
    cv::Mat dst = cv::Mat::zeros(3, 2, CV_32FC1);

    src.at<float>(0, 0) = static_cast<float>(ul[0] - pad_size[1]);
    src.at<float>(0, 1) = static_cast<float>(ul[1] - pad_size[0]);
    src.at<float>(1, 0) = static_cast<float>(br[0] + pad_size[1]);
    src.at<float>(1, 1) = static_cast<float>(br[1] + pad_size[0]);
    get_3rd_point(src.row(0), src.row(1)).copyTo(src.row(2));

    dst.row(0) = 0.f;
    dst.at<float>(1, 0) = static_cast<float>(input_res_w - 1);
    dst.at<float>(1, 1) = static_cast<float>(input_res_h - 1);
    get_3rd_point(dst.row(0), dst.row(1)).copyTo(dst.row(2));

    auto trans = cv::getAffineTransform(src, dst);
    cv::Mat dst_img = cv::Mat::zeros(input_res_h, input_res_w, CV_32FC3);
    cv::warpAffine(img, dst_img, trans, cv::Size{input_res_w, input_res_h}, cv::INTER_LINEAR);

    return std::move(dst_img);
}

float * pre_process_pose(cv::Mat & img,
                        const std::vector<bbox_t> & boxes,
                        std::vector<pt_t> & pt1,
                        std::vector<pt_t> & pt2)
{
    const int img_height = img.rows;
    const int img_width = img.cols;

    float * predict_data = (float *) malloc (boxes.size() * DEFAULT_IMG_H * DEFAULT_IMG_W * 3 * sizeof(float));
    float * p_data = predict_data;

    for (size_t i = 0; i < boxes.size(); i++)
    {
        pt_t up_left{boxes[i][0], boxes[i][1]};
        pt_t bottom_right{boxes[i][2], boxes[i][3]};

        auto box_ht = bottom_right[1] - up_left[1];
        auto box_wt = bottom_right[0] - up_left[0];
        auto scale_rate = 0.3f;

        up_left[0] = std::max(0.f, up_left[0] - box_wt * scale_rate / 2);
        up_left[1] = std::max(0.f, up_left[1] - box_ht * scale_rate / 2);

        bottom_right[0] =
            std::max(std::min(img_width - 1.f, bottom_right[0] + box_wt * scale_rate / 2), up_left[0] + 5);
        bottom_right[1] =
            std::max(std::min(img_height - 1.f, bottom_right[1] + box_ht * scale_rate / 2), up_left[1] + 5);

        auto inp = crop_box(img, up_left, bottom_right, DEFAULT_IMG_H, DEFAULT_IMG_W);
        //HWC -> CHW
        for (int row = 0; row < inp.rows; row++)
        {
            for (int col = 0; col < inp.cols; col++)
            {
                for (int c = 0; c < inp.channels(); c++)
                {
                    *(p_data + c * inp.rows * inp.cols + row * inp.cols + col) = inp.ptr<cv::Vec3f>(row, col)->val[c];
                }
            }
        }

        pt1[i] = up_left;
        pt2[i] = bottom_right;
    }

    return predict_data;
}

cv::Mat transform_box_invert_batch(cv::Mat & pt,
                                   const std::vector<pt_t> & ul, const std::vector<pt_t> & br,
                                   const int& input_res_h, const int& input_res_w,
                                   const int& output_res_h, const int& output_res_w)
{
    std::vector<pt_t> center(ul.size());
    std::vector<pt_t> size(ul.size());
    std::vector<float> len_h(ul.size());
    std::vector<float> len_w(ul.size());

    for (size_t i = 0; i < center.size(); i++)
    {
        auto & len_h_element = len_h[i];
        auto & len_w_element = len_w[i];
        len_h_element = std::numeric_limits<float>::min();
        for (size_t j = 0; j < std::tuple_size<pt_t>::value; j++)
        {
            center[i][j] = (br[i][j] - 1 - ul[i][j]) / 2;
            size[i][j] = br[i][j] - ul[i][j];
            if (j == 0)
            {
                size[i][j] *= (input_res_h * 1.f / input_res_w);
            }

            if (size[i][j] > len_h_element)
            {
                len_h_element = size[i][j];
            }
        }
        len_w_element = len_h_element *  (input_res_w * 1.f / input_res_h);
    }
    auto clamp_min_func = [](float v, float min = 0.f)
    {
        if (v < min) return min;
        return v;
    };

    cv::Mat new_point = cv::Mat::zeros(pt.dims, pt.size.p, pt.type());
    for (int i = 0; i < pt.size[0]; i++)
    {
        for (int j = 0; j < pt.size[1]; j++)
        {
            float _pt;
            _pt = pt.ptr<cv::Vec2f>(i, j)->val[0] * len_h[i] / output_res_h;
            _pt = _pt - clamp_min_func(((len_w[i] - 1) / 2 - center[i][0]));
            new_point.ptr<cv::Vec2f>(i, j)->val[0] = _pt + ul[i][0];

            _pt = pt.ptr<cv::Vec2f>(i, j)->val[1] * len_h[i] / output_res_h;
            _pt = _pt - clamp_min_func(((len_h[i] - 1) / 2 - center[i][1]));
            new_point.ptr<cv::Vec2f>(i, j)->val[1] = _pt + ul[i][1];
        }
    }

    return std::move(new_point);
}

predict_t get_predict(float * hm_data,
                      const int hm_dims[4],
                      const std::vector<pt_t> & pt1,
                      const std::vector<pt_t> & pt2,
                      const int & input_res_h,
                      const int & input_res_w)
{
    // Get Keypoint location from heatmap
    auto get_hm_data = [](float * data, const int data_dims[4], const std::array<int, 4> ele_dims)
    {
        return *(data
                 + ele_dims[0] * data_dims[1] * data_dims[2] * data_dims[3]
                 + ele_dims[1] * data_dims[2] * data_dims[3]
                 + ele_dims[2] * data_dims[3]
                 + ele_dims[3]);
    };

    cv::Mat preds(hm_dims[0], hm_dims[1],  CV_32FC2);
    cv::Mat maxval(hm_dims[0], hm_dims[1], CV_32FC1);

    for (int i = 0; i < hm_dims[0]; i++)
    {
        for (int j = 0; j < hm_dims[1]; j++)
        {
            float * start_iter = hm_data + i * hm_dims[1] * hm_dims[2] * hm_dims[3] + j * hm_dims[2] * hm_dims[3];
            auto max_element = std::max_element(start_iter, start_iter + hm_dims[2] * hm_dims[3]);
            preds.ptr<cv::Vec2f>(i, j)->val[0] = preds.ptr<cv::Vec2f>(i, j)->val[1] = std::distance(start_iter, max_element) + 1;
            maxval.at<float>(i, j) = *max_element;
        }
    }

    for (int i = 0; i < hm_dims[0]; i++)
    {
        for (int j = 0; j < hm_dims[1]; j++)
        {
            if (maxval.at<float>(i, j) < 0.)
            {
                preds.ptr<cv::Vec2f>(i, j)->val[0] = preds.ptr<cv::Vec2f>(i, j)->val[1] = 0.f;
            }
            else
            {
                preds.ptr<cv::Vec2f>(i, j)->val[0] = (size_t(preds.ptr<cv::Vec2f>(i, j)->val[0]) - 1) % hm_dims[3];
                preds.ptr<cv::Vec2f>(i, j)->val[1] = std::floor((preds.ptr<cv::Vec2f>(i, j)->val[1] - 1) / hm_dims[3]);
            }

            //Very simple post-processing step to improve performance at tight PCK thresholds
            int pX = int(std::round(preds.ptr<cv::Vec2f>(i, j)->val[0]));
            int pY = int(std::round(preds.ptr<cv::Vec2f>(i, j)->val[1]));
            if ((0 < pX)
                && (pX < (hm_dims[2] - 1))
                && (0 < pY)
                && (pY < (hm_dims[3] - 1)))
            {
                auto sign_func = [](float x)
                {
                    if (x > 0.) x = 1.f;
                    else if (x < 0.) x = -1.f;
                    return x;
                };

                float x = get_hm_data(hm_data, hm_dims, {i, j, pY, pX + 1}) - get_hm_data(hm_data, hm_dims, {i, j, pY, pX - 1});
                float y = get_hm_data(hm_data, hm_dims, {i, j, pY + 1, pX}) - get_hm_data(hm_data, hm_dims, {i, j, pY - 1, pX});
                preds.ptr<cv::Vec2f>(i, j)->val[0] += sign_func(x) * 0.25f;
                preds.ptr<cv::Vec2f>(i, j)->val[1] += sign_func(y) * 0.25f;
            }
            preds.ptr<cv::Vec2f>(i, j)->val[0] += 0.2f;
            preds.ptr<cv::Vec2f>(i, j)->val[1] += 0.2f;
        }
    }

    auto preds_tf = transform_box_invert_batch(preds, pt1, pt2, input_res_h, input_res_w, hm_dims[2], hm_dims[3]);
    return std::make_tuple(preds, preds_tf, maxval);
}

void post_process_pose(const char * image_file,
                       float * heatmap_data, int heatmap_dims[4],
                       const std::vector<pt_t>& pt1, const std::vector<pt_t>& pt2)
{
    cv::Mat preds_hm, preds_scores;
    std::tie(std::ignore, preds_hm, preds_scores) = get_predict(heatmap_data, heatmap_dims, pt1, pt2, DEFAULT_IMG_H, DEFAULT_IMG_W);
    auto preds_mean_scores = cv::mean(preds_scores.col(0));

    cv::Mat frame = cv::imread(image_file);
    for (int i = 0; i < preds_hm.rows; i++)
    {
        if (cv::mean(preds_scores.row(i)).val[0] < s_keypoint_thresh) continue;
        for (int kp_i = 0; kp_i < preds_hm.cols; kp_i++)
        {
            cv::circle(frame, cv::Point((int)preds_hm.ptr<float>(i, kp_i)[0], (int)preds_hm.ptr<float>(i, kp_i)[1]), 4, cv::Scalar(255, 255, 0), -1);
        }
    }

    cv::imwrite("Output-Keypionts.jpg", frame);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

bool tengine_predict(float * input_data, graph_t graph, const int input_dims[4], const int & num_thread, const int & loop_count)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return false;
    }

    if (set_tensor_shape(input_tensor, input_dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return false;
    }

    size_t input_data_size = (unsigned long)input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * sizeof(float);
    if (set_tensor_buffer(input_tensor, input_data, input_data_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return false;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return false;
    }

    /* run graph */
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
    double total_time = 0.;
    for (int i = 0; i < loop_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return false;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
            loop_count,
            num_thread, total_time / loop_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");
    return true;
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;

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

    /* check options */
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

    auto input_tensor = get_input_data_pose(image_file);
    int img_height = input_tensor.rows;
    int img_width = input_tensor.cols;

    // support multi-roi boxes later
    std::vector<bbox_t> boxes {{0,0, static_cast<float>(img_width - 1), static_cast<float>(img_height - 1)}};
    std::vector<pt_t> pt1, pt2;
    pt1.resize(boxes.size());
    pt2.resize(boxes.size());

    // pre-process
    float * input_data = pre_process_pose(input_tensor, boxes, pt1, pt2);
    int input_dims[] = {static_cast<int>(boxes.size()), 3, DEFAULT_IMG_H, DEFAULT_IMG_W}; // nchw

    // run prediction
    if (false == tengine_predict(input_data, graph, input_dims, num_thread, repeat_count))
    {
        fprintf(stderr, "Run model file: %s failed.\n", model_file);
        return -1;
    }

    //post process
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    int heatmap_dims[MAX_SHAPE_DIM_NUM] = {0};
    get_tensor_shape(output_tensor, heatmap_dims, MAX_SHAPE_DIM_NUM);

    post_process_pose(image_file, (float *)get_tensor_buffer(output_tensor), heatmap_dims, pt1, pt2);

    if (input_data)
    {
        free(input_data);
    }
    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
