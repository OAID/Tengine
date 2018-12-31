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
#include <iostream>
#include <atomic>

#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "prof_utils.hpp"

using namespace TEngine;

namespace caffe {

static std::atomic<uint32_t> tengine_init_count(0);

static void InitTengine(void)
{
    uint32_t prev = tengine_init_count.fetch_add(1);

    if(!prev)
        init_tengine();
}

static void ReleaseTengine(void)
{
    uint32_t prev = tengine_init_count.fetch_sub(1);

    if(prev == 1)
        release_tengine();
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const int level, const vector<string>* stages)
{
    InitTengine();

    string text_file_name = param_file;
    file_list_.push_back(text_file_name);

    prerun_already_ = false;
}

template <typename Dtype> void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename)
{
    string bin_file_name = trained_filename;
    file_list_.push_back(bin_file_name);

    graph_ = create_graph(nullptr, "caffe", file_list_[0].c_str(), trained_filename.c_str());
    if(graph_ == nullptr)
    {
        std::cerr << "Create graph failed\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return;
    }

    if(!prerun_already_)
    {
        if(prerun_graph(graph_) < 0)
        {
            std::cerr << "Prerun graph failed\n";
            return;
        }
        else
            prerun_already_ = true;
    }

    Set_input_blob();
    Set_output_blob();
}

template <typename Dtype> void Net<Dtype>::Set_input_blob()
{
    int node_number = get_graph_input_node_number(graph_);

    for(int i = 0; i < node_number; i++)
    {
        node_t node = get_graph_input_node(graph_, i);

        int tensor_number = get_node_output_number(node);

        for(int j = 0; j < tensor_number; j++)
        {
            tensor_t input_tensor = get_node_output_tensor(node, j);
            const char* tensor_name = get_tensor_name(input_tensor);

            int dims[4];
            int dim_size = get_tensor_shape(input_tensor, dims, 4);
            vector<int> shape;
            for(int i = 0; i < dim_size; i++)
                shape.push_back(dims[i]);

            // Set net input blob
            Blob<Dtype>* in_blob = new Blob<Dtype>();
            in_blob->set_name(tensor_name);
            in_blob->set_graph(graph_);
            in_blob->Reshape(shape);
            net_input_blobs_.push_back(in_blob);

            release_graph_tensor(input_tensor);
        }

        release_graph_node(node);
    }
}

template <typename Dtype> void Net<Dtype>::Set_output_blob()
{
    int node_number = get_graph_output_node_number(graph_);

    for(int i = 0; i < node_number; i++)
    {
        node_t node = get_graph_output_node(graph_, i);
        int tensor_number = get_node_output_number(node);

        for(int j = 0; j < tensor_number; j++)
        {
            tensor_t output_tensor = get_node_output_tensor(node, j);
            const char* tensor_name = get_tensor_name(output_tensor);

            int dims[4];
            int dim_size = get_tensor_shape(output_tensor, dims, 4);
            vector<int> shape;
            for(int i = 0; i < dim_size; i++)
                shape.push_back(dims[i]);

            // Set net output blob
            Blob<Dtype>* out_blob = new Blob<Dtype>();
            out_blob->set_name(tensor_name);
            out_blob->set_graph(graph_);
            out_blob->Reshape(shape);
            net_output_blobs_.push_back(out_blob);

            release_graph_tensor(output_tensor);
        }

        release_graph_node(node);
    }
}

template <typename Dtype> void Net<Dtype>::Reshape()
{
    // NOTHING NEEDS TO DO
}

template <typename Dtype> const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss)
{
    // set the shape and buffer of the input tensors
    for(unsigned int i = 0; i < net_input_blobs_.size(); i++)
    {
        Blob<Dtype>* in_blob = net_input_blobs_[i];
        const char* tensor_name = in_blob->get_name().c_str();
        tensor_t input_tensor = get_graph_tensor(graph_, tensor_name);

        int dim_size = in_blob->num_axes();
        int dims[4];

        for(int i = 0; i < dim_size; i++)
            dims[i] = in_blob->dim(i);

        set_tensor_shape(input_tensor, dims, dim_size);
        set_tensor_buffer(input_tensor, in_blob->mutable_cpu_data(), get_tensor_buffer_size(input_tensor));

        release_graph_tensor(input_tensor);
    }

    run_graph(graph_, 1);

    // set the shape and buffer of the output blobs
    for(unsigned int i = 0; i < net_output_blobs_.size(); i++)
    {
        Blob<Dtype>* out_blob = net_output_blobs_[i];
        const char* tensor_name = out_blob->get_name().c_str();
        tensor_t output_tensor = get_graph_tensor(graph_, tensor_name);

        int dims[4];
        int dim_size = get_tensor_shape(output_tensor, dims, 4);

        out_blob->set_shape(dims, dim_size);
        out_blob->set_cpu_data(( Dtype* )get_tensor_buffer(output_tensor));

        release_graph_tensor(output_tensor);
    }

    return net_output_blobs_;
}

template <typename Dtype> Net<Dtype>::~Net()
{
    postrun_graph(graph_);
    destroy_graph(graph_);

    for(auto b : net_input_blobs_)
        delete b;
    for(auto b : net_output_blobs_)
        delete b;

    ReleaseTengine();
}

INSTANTIATE_CLASS(Net);

}    // namespace caffe
