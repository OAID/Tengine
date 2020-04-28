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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/src/net.h
 * Tencent is pleased to support the open source community by making ncnn available.
 *
 * Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef __NET_H__
#define __NET_H__

#include <vector>
#include <string>
#include <mutex>
#include "tengine_c_api.h"

/* layout type, not real layout */
#define TENGINE_LAYOUT_NCHW 0
#define TENGINE_LAYOUT_NHWC 1

enum EKernelMode
{
	eKernelMode_Float32 = 0,
    eKernelMode_Int8 = 2,
    eKernelMode_Int8Perchannel,
};

namespace ncnn {
class Mat;
class Net
{
public:
    // Net initial
    Net();
    // Net clear
    ~Net();

    int load_param(const char* protopath);
    int load_model(const char* modelpath);
    
    // load model
    int load_model(context_t context, const char* model_format, const char* model_file, ...);
    // set device
    int set_device(std::string device);
    // set input shape
    int input_shape(int n, int c, int h, int w, const char* node_name);
    // input data by buffer
    int input_tensor(const float* buffer, const int buffer_size, const char* node_name);
    // output data by buffer
    int extract_tensor(float*& buffer, int& buffer_size, const char* node_name);
    // input data by tensor
    int input(std::string name, Mat& t);
    // output data by node num and tensor index
    int input_tensor(int node_index, int tensor_index, Mat& t);
    // output data by tensor
    int extract(std::string name, Mat& t);
    // output data by node num and tensor index
    int extract_tensor(int node_index, int tensor_index, Mat& t);

public:
    // set kenel mode
    static int set_kernel_mode(EKernelMode kernel_mode);
    // turn on/off wino
    static int switch_wino(bool is_open);
    // bind cpu 
    static int set_worker_cpu_list(const int* cpu_list,int num);

    // run
    int run(int block = 1);
    void dump();

public:
    graph_t graph;
    bool b_preruned;
    std::mutex net_lock_;

    // ncnn model params
    std::string ncnn_param;
    std::string ncnn_model;

private:
    // prerun
    int prerun();
};

class Mat
{
public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // image
    Mat(int w, int h, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // external packed vec
    Mat(int w, void* data, size_t elemsize, int elempack, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int elempack, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int elempack, uint8_t layout = TENGINE_LAYOUT_NCHW);

    // release
    ~Mat();
    // assign
    // Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    template <typename T> void fill(T v);
    // deep copy
    Mat clone() const;
    // reshape vec
    Mat reshape(int w) const;
    // reshape image
    Mat reshape(int w, int h) const;
    // reshape dim
    Mat reshape(int w, int h, int c) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, uint8_t layout = TENGINE_LAYOUT_NCHW);

    bool empty() const;
    size_t total() const;

    // shape only
    Mat shape() const;    

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> T* row(int y);
    template<typename T> const T* row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // convenient access float vec element
    float& operator[](size_t i);
    const float& operator[](size_t i) const;        

    enum PixelType
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB       = 1,
        PIXEL_BGR       = 2,
        PIXEL_GRAY      = 3,
        PIXEL_RGBA      = 4,
        PIXEL_BGRA      = 5,

        PIXEL_RGB2BGR   = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY  = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2RGBA  = PIXEL_RGB | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2BGRA  = PIXEL_RGB | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB   = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY  = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2RGBA  = PIXEL_BGR | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2BGRA  = PIXEL_BGR | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB  = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR  = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2RGBA = PIXEL_GRAY | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGRA = PIXEL_GRAY | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB  = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR  = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGRA = PIXEL_RGBA | (PIXEL_BGRA << PIXEL_CONVERT_SHIFT),

        PIXEL_BGRA2RGB  = PIXEL_BGRA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2BGR  = PIXEL_BGRA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2GRAY = PIXEL_BGRA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
        PIXEL_BGRA2RGBA = PIXEL_BGRA | (PIXEL_RGBA << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h);
    // convenient construct from pixel data with stride(bytes-per-row) parameter
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, int stride);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height);
    // convenient construct from pixel data and resize to specific size with stride(bytes-per-row) parameter
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type) const;
    // convenient export to pixel data with stride(bytes-per-row) parameter
    void to_pixels(unsigned char* pixels, int type, int stride) const;
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
    // convenient export to pixel data and resize to specific size with stride(bytes-per-row) parameter
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const;

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

public:
    uint8_t dim_num;

    // nchw or nhwc
    uint8_t layout;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // total num
    int elem_num;

    // point data
    void* data;

    // the dimension rank
    int dims;

    // shape
    int n;
    int c;
    int h;
    int w;

    size_t cstep;    

    // quantziation params
    std::vector<float> scales;
    std::vector<float> zero_points;
};

inline Mat::Mat() : dim_num(0), layout(0), elemsize(0), elem_num(0), data(0), n(0), c(0), h(0), w(0) 
{
}

inline Mat::Mat(int _w, size_t _elemsize, uint8_t _layout)
    : dim_num(0), layout(0), elemsize(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _elemsize, _layout);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, uint8_t _layout)
    : dim_num(0), layout(0), elemsize(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _h, _elemsize, _layout);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, uint8_t _layout)
    : dim_num(0), layout(0), elemsize(0), elem_num(0), data(0), n(0), c(0), h(0), w(0)
{
    create(_w, _h, _c, _elemsize, _layout);
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, uint8_t _layout)
    : data(_data), elemsize(_elemsize), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, uint8_t _layout)
    : data(_data), elemsize(_elemsize), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, uint8_t _layout)
    : data(_data), elemsize(_elemsize), dims(3), w(_w), h(_h), c(_c)
{
    cstep = w * h;
}


inline Mat Mat::channel(int _c)
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize);
}

inline const Mat Mat::channel(int _c) const
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize);
}

inline float* Mat::row(int y)
{
    return (float*)((unsigned char*)data + w * y * elemsize);
}

inline const float* Mat::row(int y) const
{
    return (const float*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline T* Mat::row(int y)
{
    return (T*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline const T* Mat::row(int y) const
{
    return (const T*)((unsigned char*)data + w * y * elemsize);
}

inline Mat Mat::channel_range(int _c, int channels)
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize);
}

inline const Mat Mat::channel_range(int _c, int channels) const
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize);
}

inline Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize);
}

inline const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize);
}

inline Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize);
}

inline const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize);
}

template <typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}

inline float& Mat::operator[](size_t i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](size_t i) const
{
    return ((const float*)data)[i];
}

}    // namespace tengine

#endif
