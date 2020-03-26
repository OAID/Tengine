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
 * Author: qtang@openailab.com
 */

#ifndef __TENGINE_CPP_API_H__
#define __TENGINE_CPP_API_H__

#include <vector>
#include <string>
#include <mutex>
#include "tengine_c_api.h"

/* layout type, not real layout */
#define TENGINE_LAYOUT_NCHW 0
#define TENGINE_LAYOUT_NHWC 1

namespace tengine {
class Tensor;
class Net
{
public:
    // Net initial
    Net();
    // Net clear
    ~Net();
    // load model
    int load_model(context_t context, const char* model_format, const char* model_file, ...);
    // set input shape
    int input_shape(int n, int c, int h, int w, const char* node_name);
    // input data by buffer
    int input_tensor(const float* buffer, const int buffer_size, const char* node_name);
    // output data by buffer
    int extract_tensor(float*& buffer, int& buffer_size, const char* node_name);
    // input data by tensor
    int input_tensor(std::string name, Tensor& t);
    // output data by tensor
    int extract_tensor(std::string name, Tensor& t);

    // run
    int run(int block = 1);
    void dump();

public:
    graph_t graph;
    bool b_preruned;
    std::mutex net_lock_;

private:
    // prerun
    int prerun();
};

class Tensor
{
public:
    // empty
    Tensor();
    // vec
    Tensor(int w, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // image
    Tensor(int w, int h, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // dim
    Tensor(int w, int h, int c, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // copy
    Tensor(const Tensor& m);
    // release
    ~Tensor();
    // assign
    // Tensor& operator=(const Tensor& m);
    // set all
    void fill(float v);
    template <typename T> void fill(T v);
    // deep copy
    Tensor clone() const;
    // reshape vec
    Tensor reshape(int w) const;
    // reshape image
    Tensor reshape(int w, int h) const;
    // reshape dim
    Tensor reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // allocate image
    void create(int w, int h, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // allocate dim
    void create(int w, int h, int c, size_t elem_size = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);

    bool empty() const;
    size_t total() const;

    // // convenient construct from pixel data and resize to specific size, type is RGB/BGR
    // static Tensor from_pixels_resize(const unsigned char* pixels, int type, int c, int h, int w, int target_width,
    //                                  int target_height);

    // // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip, type maybe ONNX / MXNet
    // /
    // // Caffe
    // void substract_mean_normalize(const float* mean_vals, const float* norm_vals, const int type);

public:
    uint8_t dim_num;

    // nchw or nhwc
    uint8_t layout;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elem_size;

    // total num
    int elem_num;

    // point data
    void* data;

    // shape
    int n;
    int c;
    int h;
    int w;

    // quantziation params
    std::vector<float> scales;
    std::vector<float> zero_points;
};

inline Tensor::Tensor()
    : dim_num(0), layout(0), elem_size(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
}
inline Tensor::Tensor(int _w, size_t _elem_size, uint8_t _layout)
    : dim_num(0), layout(0), elem_size(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _elem_size, _layout);
}
inline Tensor::Tensor(int _w, int _h, size_t _elem_size, uint8_t _layout)
    : dim_num(0), layout(0), elem_size(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _h, _elem_size, _layout);
}
inline Tensor::Tensor(int _w, int _h, int _c, size_t _elem_size, uint8_t _layout)
    : dim_num(0), layout(0), elem_size(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _h, _c, _elem_size, _layout);
}
}    // namespace tengine

#endif
