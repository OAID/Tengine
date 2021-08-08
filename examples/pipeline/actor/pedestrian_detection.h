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
#include <functional>
#include "pipeline/utils/box.h"
#include "pipeline/utils/profiler.h"

namespace pipeline {

class PedestrianDetection : public Node<Param<cv::Mat>, Param<std::tuple<cv::Mat, cv::Rect> > >
{
public:
    using preproc_func = typename std::function<void(const cv::Mat&, cv::Mat&)>;
    using postproc_func = typename std::function<std::vector<Box<int> >(const float*, int, int, int)>;

    PedestrianDetection(std::string model_path, size_t thread = 2, int w = 300, int h = 300)
    {
        m_thread_num = thread;
        m_input = cv::Mat(h, w, CV_32FC3);

        // build preproc
        m_preproc = [](const cv::Mat& in, cv::Mat& out) -> void {
            cv::Mat buf(out.rows, out.cols, CV_8UC3);
            cv::resize(in, buf, buf.size());
            cv::cvtColor(buf, buf, CV_BGR2RGB);

            buf.convertTo(buf, CV_32FC3);

            const float mean[3] = {127.5f, 127.5f, 127.5f};
            const float scale[3] = {0.007843f, 0.007843f, 0.007843f};

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

        // build postproc
        m_postproc = [](const float* outdata, int num, int raw_h, int raw_w) -> std::vector<Box<int> > {
            const char* class_names[] = {
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train",
                "tvmonitor"};

            const int max_num = num;
            Box<int> boxes[max_num];
            for (int i = 0; i < max_num; ++i)
            {
                boxes[i] = {0};
            }
            int box_count = 0;

            fprintf(stderr, "detect result num: %d \n", num);
            for (int i = 0; i < num; i++)
            {
                if (outdata[1] >= 0.5f)
                {
                    Box<int> box;

                    box.class_idx = outdata[0];
                    box.score = outdata[1];
                    box.x0 = outdata[2] * raw_w;
                    box.y0 = outdata[3] * raw_h;
                    box.x1 = outdata[4] * raw_w;
                    box.y1 = outdata[5] * raw_h;

                    boxes[box_count] = box;
                    box_count++;

                    fprintf(stderr, "%s\t:%.1f%%\n", class_names[box.class_idx],
                            box.score * 100);
                    fprintf(stderr, "BOX:( %d , %d ),( %d , %d )\n", box.x0, box.y0, box.x1,
                            box.y1);
                }
                outdata += 6;
            }

            Box<int> max = {0};
            for (int i = 0; i < box_count; i++)
            {
                if (boxes[i].score > max.score)
                {
                    max = boxes[i];
                }
            }

            std::vector<Box<int> > ret = {max};
            return ret;
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
        Profiler prof("pedestrian_detection");
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
        tensor_t output_tensor = get_graph_output_tensor(m_graph, 0, 0); //"detection_out"
        int out_dim[4];
        get_tensor_shape(output_tensor, out_dim, 4);
        float* output_data = (float*)get_tensor_buffer(output_tensor);

        /* postprocess*/
        fprintf(stdout, "out shape [%d %d %d %d]\n", out_dim[0], out_dim[1],
                out_dim[2], out_dim[3]);
        auto results = m_postproc(output_data, out_dim[1], mat.rows, mat.cols);

        prof.dot();

        if (not results.empty())
        {
            auto& result = results[0];
            cv::Rect rect(std::max(0, result.x0), std::max(0, result.y0),
                          result.x1 - result.x0, result.y1 - result.y0);
            rect.width = std::min(rect.width, mat.cols - rect.x - 1);
            rect.height = std::min(rect.height, mat.rows - rect.y - 1);
            cv::rectangle(mat, rect, cv::Scalar(255, 255, 255), 3);

            output<0>()->try_push(std::move(std::make_tuple(mat, rect)));
        }
        else
        {
            output<0>()->try_push(std::move(std::make_tuple(mat, cv::Rect(0, 0, 0, 0))));
        }
    }

    ~PedestrianDetection()
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
    postproc_func m_postproc;

    cv::Mat m_input;
    size_t m_thread_num;
};

} // namespace pipeline
