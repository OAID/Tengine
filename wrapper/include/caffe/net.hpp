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
#ifndef __CAFFE_NET_HPP__
#define __CAFFE_NET_HPP__

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe.pb.h"
#include "tengine_c_api.h"

#define Net Net_wrap

using namespace std;

namespace caffe {

template <typename Dtype> class Net
{
public:
    explicit Net(const string& param_file, Phase phase, const int level = 0, const vector<string>* stages = NULL);
    ~Net();

    void CopyTrainedLayersFrom(const string trained_filename);

    // Run Forward and return the result
    const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);

    // Reshape all layers from bottom to top
    void Reshape();

    // Input and output blob numbers
    int num_inputs() const
    {
        return net_input_blobs_.size();
    }
    int num_outputs() const
    {
        return net_output_blobs_.size();
    }

    // Get input and output blobs
    const vector<Blob<Dtype>*>& input_blobs() const
    {
        return net_input_blobs_;
    }

    const vector<Blob<Dtype>*>& output_blobs() const
    {
        return net_output_blobs_;
    }

protected:
    vector<string> file_list_;    // model file list
    bool prerun_already_;

    graph_t graph_;    // pointer of graph executor

    vector<Blob<Dtype>*> net_input_blobs_;
    vector<Blob<Dtype>*> net_output_blobs_;

    void Set_input_blob();
    void Set_output_blob();
};

}    // namespace caffe

#endif    // __CAFFE_NET_HPP__
