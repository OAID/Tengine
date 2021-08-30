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
#include "pipeline/utils/feature.h"
#include "pipeline/utils/profiler.h"

namespace pipeline {

#define HARD_NMS     (1)
#define BLENDING_NMS (2) /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

class FaceFeature : public Node<Param<std::tuple<cv::Mat, std::vector<Feature> > >, Param<Feature> >
{
public:
    using preproc_func = typename std::function<void(const cv::Mat&, cv::Mat&)>;

    FaceFeature(std::string model_path, size_t thread = 2, int w = 110, int h = 110)
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

            float mean[3] = {104.007f, 116.669f, 122.679f};
            float scale[3] = {1., 1., 1.};

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

        const int size = static_cast<int>(m_input.cols * m_input.rows * m_input.elemSize());
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
        std::vector<Feature> features;

        std::vector<Feature> out;
        std::tuple<cv::Mat, std::vector<Feature> > inp;
        auto suc = input<0>()->pop(inp);
        if (not suc)
        {
            return;
        }

        std::tie(mat, features) = inp;

        auto get_bbox = [&](const Feature& feature) -> cv::Rect2f {
            /* get landmark bbox */
            float l = FLT_MAX, r = FLT_MIN, t = FLT_MAX, b = FLT_MIN;
            auto& data = feature.data;
            for (int i = 0; i < data.size() / 2; ++i)
            {
                int idx = i * 2;
                if (l > data[idx])
                {
                    l = data[idx];
                }
                if (r < data[idx])
                {
                    r = data[idx];
                }

                if (t > data[idx + 1])
                {
                    t = data[idx + 1];
                }

                if (b < data[idx + 1])
                {
                    b = data[idx + 1];
                }
            }

            l = std::max(0.f, l);
            t = std::max(0.f, t);
            r = std::min(r, mat.cols * 1.f);
            b = std::min(b, mat.rows * 1.f);

            return cv::Rect2f(l, t, r - l, b - t);
        };

        for (auto feature : features)
        {
            auto rect = get_bbox(feature);
            cv::Mat crop = mat(rect);

            /* prepare process input data, set the data mem to input tensor */
            Profiler prof("face_feature");
            prof.dot();
            m_preproc(crop, m_input);
            prof.dot();

            if (run_graph(m_graph, 1) < 0)
            {
                fprintf(stderr, "Run graph failed\n");
                return;
            }
            prof.dot();

            /* process the landmark result */
            tensor_t output_tensor = get_graph_output_tensor(m_graph, 0, 0);
            float* data = (float*)(get_tensor_buffer(output_tensor));
            const int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

            fprintf(stdout, "write feature %f, len %d\n ", data[0], data_size);

            Feature f;
            f.data = {data, data + data_size};
            output<0>()->try_push(std::move(f));
        }
        return;
    }

    ~FaceFeature()
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
