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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <atomic>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "prof_utils.hpp"

using namespace TEngine;

namespace caffe {

static std::atomic<uint32_t> tengine_init_count(0);

static void InitTengine(void) {
  uint32_t prev = tengine_init_count.fetch_add(1);

  if (!prev) init_tengine_library();
}

static void ReleaseTengine(void) {
  uint32_t prev = tengine_init_count.fetch_sub(1);

  if (prev == 1) release_tengine_library();
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const int level,
                const vector<string>* stages) {
  InitTengine();

  string text_file_name = param_file;
  file_list_.push_back(text_file_name);

  prerun_already_ = false;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  string bin_file_name = trained_filename;
  file_list_.push_back(bin_file_name);

  model_name_ = trained_filename;

  // Load model and create static graph
  if (load_model(model_name_.c_str(), "caffe", file_list_[0].c_str(),
                 trained_filename.c_str()) < 0) {
    std::cerr << "Load model failed\n";
    return;
  }

  // Create runtime graph
  graph_ = create_runtime_graph("graph0", model_name_.c_str(), NULL);
  if (!check_graph_valid(graph_)) {
    std::cerr << "Create graph0 failed\n";
    return;
  }

  if (infer_shape(graph_) < 0) {
    std::cerr << "Infer shape failed\n";
    return;
  }

  Set_input_blob();
  Set_output_blob();
}

template <typename Dtype>
void Net<Dtype>::Set_input_blob() {
  int node_number = get_input_node_number(graph_);

  for (int i = 0; i < node_number; i++) {
    const char* node_name = get_input_node_name(graph_, i);
    int tensor_number = get_node_output_number(graph_, node_name);

    for (int j = 0; j < tensor_number; j++) {
      const char* tensor_name = get_node_output_tensor(graph_, node_name, j);
      tensor_t input_tensor = get_graph_tensor(graph_, tensor_name);

      int dims[4];
      get_tensor_shape(input_tensor, dims, 4);

      // Set net input blob
      Blob<Dtype>* in_blob = new Blob<Dtype>();
      in_blob->set_name(tensor_name);
      in_blob->set_graph(graph_);
      in_blob->Reshape(dims[0], dims[1], dims[2], dims[3]);
      net_input_blobs_.push_back(in_blob);

      put_graph_tensor(input_tensor);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::Set_output_blob() {
  int node_number = get_output_node_number(graph_);

  for (int i = 0; i < node_number; i++) {
    const char* node_name = get_output_node_name(graph_, i);
    int tensor_number = get_node_output_number(graph_, node_name);

    for (int j = 0; j < tensor_number; j++) {
      const char* tensor_name = get_node_output_tensor(graph_, node_name, j);
      tensor_t output_tensor = get_graph_tensor(graph_, tensor_name);

      int dims[4];
      get_tensor_shape(output_tensor, dims, 4);

      // Set net output blob
      Blob<Dtype>* out_blob = new Blob<Dtype>();
      out_blob->set_name(tensor_name);
      out_blob->set_graph(graph_);
      out_blob->Reshape(dims[0], dims[1], dims[2], dims[3]);
      net_output_blobs_.push_back(out_blob);

      put_graph_tensor(output_tensor);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  if (prerun_already_ == true) {
    postrun_graph(graph_);
  }

  if (prerun_graph(graph_) < 0)
    std::cerr << "Net reshape failed\n";
  else
    prerun_already_ = true;

  for (unsigned int i = 0; i < net_output_blobs_.size(); i++) {
    Blob<Dtype>* out_blob = net_output_blobs_[i];
    const char* tensor_name = out_blob->get_name().c_str();
    tensor_t output_tensor = get_graph_tensor(graph_, tensor_name);

    int dims[4];
    get_tensor_shape(output_tensor, dims, 4);
    out_blob->Reshape(dims[0], dims[1], dims[2], dims[3]);

    put_graph_tensor(output_tensor);
  }
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  // int repeat_count=100;
  // unsigned long start_time = get_cur_time();

  // for(int i=0; i < repeat_count; i++)
  run_graph(graph_, 1);

  // unsigned long end_time = get_cur_time();

  // printf("Repeat [%d] times %.2f per RUN, used [%lu] us\n", repeat_count,
  //       1.0f*(end_time-start_time)/repeat_count, end_time-start_time);

  return net_output_blobs_;
}

template <typename Dtype>
Net<Dtype>::~Net() {
  postrun_graph(graph_);
  destroy_runtime_graph(graph_);
  remove_model(model_name_.c_str());

  for (auto b : net_input_blobs_) delete b;
  for (auto b : net_output_blobs_) delete b;

  ReleaseTengine();
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
