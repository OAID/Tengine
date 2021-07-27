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
#include <sys/time.h>
#include <functional>
#include "pipeline/utils/box.h"

namespace pipe {

class PedestrianDetection : public Node<Param<cv::Mat>, Param<std::tuple<cv::Mat, cv::Rect>>> {
public:
  using preproc_func = typename std::function<void (const cv::Mat&, cv::Mat&)>;
  using postproc_func = typename std::function<std::vector<Box<int>>(const float*, int)>;

  PedestrianDetection(std::string model_path, preproc_func preproc, postproc_func postproc, size_t thread = 2, int w = 300, int h = 300) {
    m_thread_num = thread;
    m_input = cv::Mat(h, w, CV_32FC3);
    m_preproc = preproc;
    m_postproc = postproc;

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
    if (m_graph == nullptr) {
      fprintf(stderr, "create graph failed\n");
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
    if (input_tensor == NULL) {
      fprintf(stderr, "Get input tensor failed\n");
      return;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0) {
      fprintf(stderr, "Set input tensor shape failed\n");
      return;
    }

    const int size = m_input.cols * m_input.rows * m_input.elemSize();
    fprintf(stdout, "tensor_buffer size %d\n", size);
    if (set_tensor_buffer(input_tensor, (void*)(m_input.data), size) < 0) {
      fprintf(stderr, "Set input tensor buffer failed\n");
      return;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(m_graph, opt) < 0) {
      fprintf(stderr, "Prerun graph failed\n");
      return;
    }

    fprintf(stdout, "init success\n");
  }

  void exec() override {
    cv::Mat mat;
    auto suc = input<0>()->pop(mat);
    if (not suc or mat.empty()) {
      return;
    }

    auto get_current_time = []() -> double {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    };

    /* prepare process input data, set the data mem to input tensor */
    auto time1 = get_current_time();

    fprintf(stdout, "preproc begin\n");
    m_preproc(mat, m_input);
    fprintf(stdout, "preproc end\n");

    auto time2 = get_current_time();

    if (run_graph(m_graph, 1) < 0) {
      fprintf(stderr, "Run graph failed\n");
      return;
    }

    auto time3 = get_current_time();

    /* process the detection result */
    tensor_t output_tensor =
        get_graph_output_tensor(m_graph, 0, 0); //"detection_out"
    int out_dim[4];
    get_tensor_shape(output_tensor, out_dim, 4);
    float *output_data = (float *)get_tensor_buffer(output_tensor);

    /* postprocess*/
    fprintf(stdout, "out shape [%d %d %d %d]\n", out_dim[0], out_dim[1],
            out_dim[2], out_dim[3]);
    auto results = m_postproc(output_data, out_dim[1]);

    auto time4 = get_current_time();

    fprintf(stdout, "preproc %.2f,  inference %.2f,  postproc %.2f \n",
            time2 - time1, time3 - time2, time4 - time3);

    if (not results.empty()) {
      auto& result = results[0];
      cv::Rect rect(std::max(0, result.x0), std::max(0, result.y0),
              result.x1 - result.x0, result.y1 - result.y0);
      rect.width = std::min(rect.width, mat.cols - rect.x - 1);
      rect.height = std::min(rect.height, mat.rows - rect.y - 1);
      cv::rectangle(mat, rect, cv::Scalar(255, 255, 255), 3);

      output<0>()->try_push(std::move(std::make_tuple(mat, rect)));
    } else {
      output<0>()->try_push(std::move(std::make_tuple(mat, cv::Rect(0, 0, 0, 0))));
    }
  }

  ~PedestrianDetection() {
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

} // namespace pipe
