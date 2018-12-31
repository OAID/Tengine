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
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width)
{
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    Reshape(shape);
}

template <typename Dtype> void Blob<Dtype>::Reshape(const vector<int>& shape)
{
    int dim_size = shape.size();
    int* dims = new int[dim_size];
    shape_.resize(dim_size);
    count_ = 1;

    for(int i = 0; i < dim_size; ++i)
    {
        if(shape[i] < 0)
            std::cerr << "Shape number error\n";
        else
        {
            shape_[i] = shape[i];
            dims[i] = shape[i];
            count_ *= shape[i];
        }
    }

    if(count_ > capacity_)
    {
        if(!capacity_)
            data_ = malloc(sizeof(Dtype) * count_);
        else
            data_ = realloc(data_, sizeof(Dtype) * count_);

        capacity_ = count_;
    }

    delete[] dims;
}

template <typename Dtype> const Dtype* Blob<Dtype>::cpu_data() const
{
    if(!data_)
    {
        std::cerr << "Get blob cpu data failed\n";
        return nullptr;
    }

    return ( const Dtype* )data_;
}

template <typename Dtype> Dtype* Blob<Dtype>::mutable_cpu_data()
{
    if(!data_)
    {
        std::cerr << "Get blob mutable cpu data failed\n";
        return nullptr;
    }
    return static_cast<Dtype*>(data_);
}

template <typename Dtype> void Blob<Dtype>::set_cpu_data(Dtype* data)
{
    if(!data)
    {
        std::cerr << "Set blob cpu data failed\n";
    }

    if(data_ && !external_mem_)
    {
        free(data_);
    }

    data_ = ( void* )data;
    external_mem_ = true;
}

template <typename Dtype> void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape)
{
    if(reshape)
    {
        vector<int> shape;
        if(proto.has_num() || proto.has_channels() || proto.has_height() || proto.has_width())
        {
            shape.resize(4);
            shape[0] = proto.num();
            shape[1] = proto.channels();
            shape[2] = proto.height();
            shape[3] = proto.width();
        }
        else
        {
            shape.resize(proto.shape().dim_size());
            for(int i = 0; i < proto.shape().dim_size(); ++i)
                shape[i] = proto.shape().dim(i);
        }
        Reshape(shape);
    }
    else
    {
        if(data_)
            free(data_);

        data_ = malloc(sizeof(Dtype) * count_);
    }

    Dtype* data = ( Dtype* )data_;

    if(proto.double_data_size() > 0)
    {
        for(int i = 0; i < proto.double_data_size(); ++i)
            data[i] = proto.double_data(i);
    }
    else
    {
        for(int i = 0; i < proto.data_size(); ++i)
            data[i] = proto.data(i);
    }
}

INSTANTIATE_CLASS(Blob);

}    // namespace caffe
