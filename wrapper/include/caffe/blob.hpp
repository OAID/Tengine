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
#ifndef __CAFFE_BLOB_HPP__
#define __CAFFE_BLOB_HPP__

#include <iostream>
#include <vector>
#include "caffe.pb.h"
#include "tengine_c_api.h"

#define Blob Blob_wrap

using namespace std;

namespace caffe {

template <typename Dtype> class Blob
{
public:
    Blob() : data_(nullptr), count_(0), capacity_(0)
    {
        external_mem_ = false;
    }
    ~Blob()
    {
        if(data_ && !external_mem_)
            free(data_);
    }

    void FromProto(const BlobProto& proto, bool reshape = true);
    void Reshape(const int num, const int channels, const int height, const int width);
    void Reshape(const vector<int>& shape);

    const Dtype* cpu_data() const;
    Dtype* mutable_cpu_data();
    void set_cpu_data(Dtype* data);
    void set_shape(int* dims, int dim_size)
    {
        shape_.resize(dim_size);
        for(int i = 0; i < dim_size; i++)
            shape_[i] = dims[i];
    }

    int dim(int i) const
    {
        return shape_.at(i);
    }
    int num() const
    {
        return shape_.at(0);
    }
    int channels() const
    {
        return shape_.at(1);
    }
    int height() const
    {
        return shape_.at(2);
    }
    int width() const
    {
        return shape_.at(3);
    }

    const vector<int>& shape() const
    {
        return shape_;
    }
    int shape(int index) const
    {
        return shape_[CanonicalAxisIndex(index)];
    }
    int num_axes() const
    {
        return shape_.size();
    }

    void set_name(string name)
    {
        name_ = name;
    }
    string get_name()
    {
        return name_;
    }

    void set_graph(graph_t graph)
    {
        graph_ = graph;
    }

    int count() const
    {
        return count(0);
    }
    int count(int start_axis) const
    {
        return count(start_axis, num_axes());
    }
    int count(int start_axis, int end_axis) const
    {
        if(start_axis > end_axis || start_axis < 0 || end_axis < 0 || start_axis > num_axes() || end_axis > num_axes())
        {
            std::cerr << "parameter out of range\n";
            return 0;
        }

        int count = 1;
        for(int i = start_axis; i < end_axis; ++i)
            count *= shape(i);
        return count;
    }

    int CanonicalAxisIndex(int axis_index) const
    {
        if(axis_index < -num_axes() || axis_index >= num_axes())
        {
            std::cerr << "axis " << axis_index << " out of range for " << num_axes() << "\n";
            return 0;
        }

        if(axis_index < 0)
            return axis_index + num_axes();

        return axis_index;
    }

protected:
    string name_;    // tensor name
    vector<int> shape_;
    void* data_;
    bool external_mem_;
    int count_;
    int capacity_;
    graph_t graph_;    // pointer of graph executor

};    // class Blob

}    // namespace caffe

#endif    // __CAFFE_BLOB_HPP__
