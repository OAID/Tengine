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
 * Copyright (c) 2021
 * Author: tpoisonooo
 */
#pragma once
#include "pipeline/graph/node.h"
#include "tengine/c_api.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 4
#include <opencv2/imgproc/types_c.h>
#endif
#include <sys/time.h>
#include <functional>
#include "pipeline/utils/box.h"
#include "pipeline/utils/profiler.h"

namespace pipeline {

#define HARD_NMS     (1)
#define BLENDING_NMS (2) /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

class FaceDetection : public Node<Param<cv::Mat>, Param<std::tuple<cv::Mat, std::vector<cv::Rect> > > >
{
public:
    using preproc_func = typename std::function<void(const cv::Mat&, cv::Mat&)>;

    FaceDetection(std::string model_path, size_t thread = 2, int w = 320, int h = 240)
    {
        m_tensor_in_w = w;
        m_tensor_in_h = h;
        m_thread_num = thread;
        m_input = cv::Mat(h, w, CV_32FC3);

        // build preproc
        m_preproc = [](const cv::Mat& in, cv::Mat& out) -> void {
            cv::Mat buf(out.rows, out.cols, CV_8UC3);
            cv::resize(in, buf, buf.size());
            cv::cvtColor(buf, buf, CV_BGR2RGB);

            buf.convertTo(buf, CV_32FC3);

            float mean[3] = {127.f, 127.f, 127.f};
            float scale[3] = {1.0f / 128, 1.0f / 128, 1.0f / 128};

            float* img_data = reinterpret_cast<float*>(buf.data);
            float* out_ptr = reinterpret_cast<float*>(out.data);
            /* nhwc to nchw */
            for (int h = 0; h < out.rows; h++)
            {
                for (int w = 0; w < out.cols; w++)
                {
#pragma unroll(3)
                    for (int c = 0; c < 3; c++)
                    {
                        int in_index = h * out.cols * 3 + w * 3 + c;
                        int out_index = c * out.cols * out.rows + h * out.cols + w;
                        out_ptr[out_index] = (img_data[in_index] - mean[c]) * scale[c];
                    }
                }
            }
        };

        /* inital tengine */
        init_tengine();
        fprintf(stderr, "tengine-lite library version: %s\n",
                get_tengine_version());

        m_ctx = create_context("ctx", 1);

        auto ret = set_context_device(m_ctx, "dev", NULL, 0);
        if (0 != ret)
        {
            fprintf(stderr, "set context device failed, skip\n");
        }
        m_graph = create_graph(NULL, "tengine", model_path.c_str());
        if (m_graph == nullptr)
        {
            fprintf(stderr, "create graph failed, check model path\n");
            return;
        }

        /* set runtime options */
        struct options opt;
        opt.num_thread = m_thread_num;
        opt.cluster = TENGINE_CLUSTER_ALL;
        opt.precision = TENGINE_MODE_FP32;
        opt.affinity = 0;

        /* set the input shape to initial the graph, and prerun graph to infer shape
     */
        int dims[] = {1, 3, m_input.rows, m_input.cols}; // nchw

        tensor_t input_tensor = get_graph_input_tensor(m_graph, 0, 0);
        if (input_tensor == NULL)
        {
            fprintf(stderr, "Get input tensor failed\n");
            return;
        }

        if (set_tensor_shape(input_tensor, dims, 4) < 0)
        {
            fprintf(stderr, "Set input tensor shape failed\n");
            return;
        }

        const int size = m_input.cols * m_input.rows * m_input.elemSize();
        fprintf(stdout, "tensor_buffer size %d\n", size);
        if (set_tensor_buffer(input_tensor, (void*)(m_input.data), size) < 0)
        {
            fprintf(stderr, "Set input tensor buffer failed\n");
            return;
        }

        /* prerun graph, set work options(num_thread, cluster, precision) */
        if (prerun_graph_multithread(m_graph, opt) < 0)
        {
            fprintf(stderr, "Prerun graph failed\n");
            return;
        }

        fprintf(stdout, "init success\n");
    }

    void exec() override
    {
        cv::Mat mat;
        auto suc = input<0>()->pop(mat);
        if (not suc or mat.empty())
        {
            return;
        }

        /* prepare process input data, set the data mem to input tensor */
        Profiler prof("face_detection");
        prof.dot();
        m_preproc(mat, m_input);
        prof.dot();

        if (run_graph(m_graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return;
        }
        prof.dot();

        /* process the detection result */
        tensor_t boxs_tensor = get_graph_output_tensor(m_graph, 0, 0);
        tensor_t scores_tensor = get_graph_output_tensor(m_graph, 1, 0);

        float* boxs_data = (float*)get_tensor_buffer(boxs_tensor);
        float* scores_data = (float*)get_tensor_buffer(scores_tensor);

        auto results = postprocess(mat, boxs_data, scores_data);
        prof.dot();

        std::vector<cv::Rect> rects;
        if (not results.empty())
        {
            auto result = std::max_element(results.begin(), results.end(), [](Box<float> a, Box<float> b) -> bool {
                return a.score < b.score;
            });

            cv::Rect rect(std::max(0.f, result->x0), std::max(0.f, result->y0),
                          result->x1 - std::max(0.f, result->x0), result->y1 - std::max(0.f, result->y0));
            rect.width = std::min(rect.width, mat.cols - rect.x - 1);
            rect.height = std::min(rect.height, mat.rows - rect.y - 1);

            // cv::rectangle(mat, rect, cv::Scalar(255, 255, 255), 3);
            rects.emplace_back(rect);

            output<0>()->try_push(std::move(std::make_tuple(mat, rects)));
        }
        else
        {
            output<0>()->try_push(std::move(std::make_tuple(mat, rects)));
        }
        return;
    }

    void nms(std::vector<Box<float> >& input, std::vector<Box<float> >& output, int type = BLENDING_NMS)
    {
#define NUM_FEATUREMAP (4)
#define clip(x, y)     (x < 0 ? 0 : (x > y ? y : x))

        const float iou_threshold = 0.3f;
        std::sort(input.begin(), input.end(), [](const Box<float>& a, const Box<float>& b) { return a.score > b.score; });

        int box_num = input.size();

        std::vector<int> merged(box_num, 0);

        for (int i = 0; i < box_num; i++)
        {
            if (merged[i])
                continue;
            std::vector<Box<float> > buf;

            buf.emplace_back(input[i]);
            merged[i] = 1;

            float h0 = input[i].y1 - input[i].y0 + 1;
            float w0 = input[i].x1 - input[i].x0 + 1;

            float area0 = h0 * w0;

            for (int j = i + 1; j < box_num; j++)
            {
                if (merged[j])
                    continue;

                float inner_x0 = input[i].x0 > input[j].x0 ? input[i].x0 : input[j].x0;
                float inner_y0 = input[i].y0 > input[j].y0 ? input[i].y0 : input[j].y0;

                float inner_x1 = input[i].x1 < input[j].x1 ? input[i].x1 : input[j].x1;
                float inner_y1 = input[i].y1 < input[j].y1 ? input[i].y1 : input[j].y1;

                float inner_h = inner_y1 - inner_y0 + 1;
                float inner_w = inner_x1 - inner_x0 + 1;

                if (inner_h <= 0 || inner_w <= 0)
                    continue;

                float inner_area = inner_h * inner_w;

                float h1 = input[j].y1 - input[j].y0 + 1;
                float w1 = input[j].x1 - input[j].x0 + 1;

                float area1 = h1 * w1;

                float score;

                score = inner_area / (area0 + area1 - inner_area);

                if (score > iou_threshold)
                {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }
            switch (type)
            {
            case HARD_NMS:
            {
                output.push_back(buf[0]);
                break;
            }
            case BLENDING_NMS:
            {
                float total = 0;
                for (int i = 0; i < buf.size(); i++)
                {
                    total += exp(buf[i].score);
                }
                Box<float> rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < buf.size(); i++)
                {
                    float rate = exp(buf[i].score) / total;
                    rects.x0 += buf[i].x0 * rate;
                    rects.y0 += buf[i].y0 * rate;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default:
            {
                fprintf(stderr, "wrong type of nms.");
                exit(-1);
            }
            }
        }
    }

    std::vector<Box<float> > postprocess(cv::Mat& m, float* boxs_data, float* scores_data)
    {
        const int image_h = m.rows;
        const int image_w = m.cols;

        const std::vector<std::vector<float> > min_boxes = {
            {10.0f, 16.0f, 24.0f}, {32.0f, 48.0f}, {64.0f, 96.0f}, {128.0f, 192.0f, 256.0f}};
        std::vector<std::vector<float> > shrinkage_size;
        std::vector<std::vector<float> > priors = {};
        std::vector<std::vector<float> > featuremap_size;
        const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
        std::vector<int> w_h_list = {m_tensor_in_w, m_tensor_in_h};

        for (auto size : w_h_list)
        {
            std::vector<float> fm_item;
            for (float stride : strides)
            {
                fm_item.emplace_back(ceil(size / stride));
            }
            featuremap_size.emplace_back(fm_item);
        }

        for (auto size : w_h_list)
        {
            shrinkage_size.push_back(strides);
        }
        /* generate prior anchors */
        for (int index = 0; index < NUM_FEATUREMAP; index++)
        {
            float scale_w = m_tensor_in_w / shrinkage_size[0][index];
            float scale_h = m_tensor_in_h / shrinkage_size[1][index];
            for (int j = 0; j < featuremap_size[1][index]; j++)
            {
                for (int i = 0; i < featuremap_size[0][index]; i++)
                {
                    float x_center = (i + 0.5) / scale_w;
                    float y_center = (j + 0.5) / scale_h;

                    for (float k : min_boxes[index])
                    {
                        float w = k / m_tensor_in_w;
                        float h = k / m_tensor_in_h;
                        priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                    }
                }
            }
        }
        /* generate prior anchors finished */
        std::vector<Box<float> > bbox_collection;
        const int num_anchors = priors.size();
        const float score_threshold = 0.7f;
        const float center_variance = 0.1f;
        const float size_variance = 0.2f;
        for (int i = 0; i < num_anchors; i++)
        {
            if (scores_data[i * 2 + 1] > score_threshold)
            {
                Box<float> rects;
                float x_center = boxs_data[i * 4] * center_variance * priors[i][2] + priors[i][0];
                float y_center = boxs_data[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
                float w = exp(boxs_data[i * 4 + 2] * size_variance) * priors[i][2];
                float h = exp(boxs_data[i * 4 + 3] * size_variance) * priors[i][3];

                rects.x0 = clip(x_center - w / 2.0, 1) * image_w;
                rects.y0 = clip(y_center - h / 2.0, 1) * image_h;
                rects.x1 = clip(x_center + w / 2.0, 1) * image_w;
                rects.y1 = clip(y_center + h / 2.0, 1) * image_h;
                rects.score = clip(scores_data[i * 2 + 1], 1);
                bbox_collection.emplace_back(rects);
            }
        }

        std::vector<Box<float> > face_list;
        nms(bbox_collection, face_list);

        fprintf(stderr, "detected face num: %ld\n", face_list.size());
        return face_list;
    }

    ~FaceDetection()
    {
        /* release tengine */
        postrun_graph(m_graph);
        destroy_graph(m_graph);
        release_tengine();
    }

private:
    graph_t m_graph;
    context_t m_ctx;
    preproc_func m_preproc;

    cv::Mat m_input;

    int m_tensor_in_w;
    int m_tensor_in_h;
    size_t m_thread_num;
};

} // namespace pipeline
