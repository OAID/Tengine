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
 * Copyright (c) 2020, OEPN AI LAB
 * Author: qtang@openailab.com
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#include "tengine_cpp_api.h"

namespace tengine {

Net::Net()
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    graph = nullptr;
    b_preruned = false;

    opt.num_thread = 1;
    opt.precision = TENGINE_MODE_FP32;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.affinity = 0;
}

Net::~Net()
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    postrun_graph(graph);
    destroy_graph(graph);

    graph = nullptr;
}

int Net::load_model(context_t context, const char* model_format, const char* model_file, ...)
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    graph = create_graph(nullptr, model_format, model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    return 0;
}

int Net::set_device(const std::string& device) const
{
    if(!device.empty())
    {
        set_graph_device(graph, device.c_str());
    }

    return 0;
}

int Net::input_shape(const std::string& tensor_name, int n, int c, int h, int w) const
{
    // printf("tengine cpp api : %s name : %s %d:%d:%d:%d\n", __FUNCTION__, tensor_name, n, c, h, w);
    int dims[4];

    dims[0] = n;
    dims[1] = c;
    dims[2] = h;
    dims[3] = w;
    tensor_t tensor = get_graph_tensor(graph, tensor_name.c_str());
    if(tensor == nullptr)
    {
        std::printf("Cannot find tensor name: %s\n", tensor_name.c_str());
        return -1;
    }

    int ret = set_tensor_shape(tensor, dims, 4);

    return ret;
}

int Net::prerun() const
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    int ret = prerun_graph_multithread(graph, opt);
    if(ret < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    return 0;
}

int Net::run(int block)
{
    if (!b_preruned)
    {
        int ret = prerun();
        if(ret == 0)
            b_preruned = true;
        else
        {
            return ret;
        }
    }
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    int ret = run_graph(graph, block);

    return ret;
}

void Net::dump() const
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    dump_graph(graph);
}

int Net::input_tensor(const std::string& tensor_name, Tensor& t) const
{
    /* set shape of input tensor */
    if (input_shape(tensor_name, t.n, t.c, t.h, t.w) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__,name.c_str());
    tensor_t tensor = get_graph_tensor(graph, tensor_name.c_str());
    if(tensor == nullptr)
    {
        std::printf("Cannot find tensor name: %s\n", tensor_name.c_str());
        return -1;
    }

    /* bind data buffer of input tensor */
    if (set_tensor_buffer(tensor, ( void* )t.data, t.total()) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    return 0;
}

int Net::input_tensor(const std::string& tensor_name, const float* buffer, const int buffer_size) const
{
    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__, tensor_name);
    tensor_t tensor = get_graph_tensor(graph, tensor_name.c_str());
    if(tensor == nullptr)
    {
        std::printf("Cannot find tensor name: %s\n", tensor_name.c_str());
        return -1;
    }
    int ret = set_tensor_buffer(tensor, ( void* )buffer, buffer_size);
    if(ret < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    return 0;
}

int Net::extract_tensor(const std::string& tensor_name, Tensor& t) const
{
    tensor_t tensor = get_graph_tensor(graph, tensor_name.c_str());
    if(tensor == nullptr)
    {
        std::printf("Cannot find output tensor , tensor_name: %s \n", tensor_name.c_str());
        return -1;
    }
    int dims[4] = {0};
    int dim_num = 4;
    dim_num = get_tensor_shape(tensor, dims, dim_num);
    if(dim_num < 0)
    {
        std::printf("Get tensor shape failed\n");
        return -1;
    }

    int layout = get_tensor_layout(tensor);
    if(layout == TENGINE_LAYOUT_NCHW)
    {
        // printf("tengine cpp api : %s dims: n %d, c %d, h %d, w %d\n", __FUNCTION__, dims[0], dims[1], dims[2],
        // dims[3]); Mat m;
        if(dim_num == 4)
            t.create(dims[0], dims[1], dims[2], dims[3], 4, TENGINE_LAYOUT_NCHW);
        else
        {
            std::printf("Get tensor dim num is not 4, failed\n");
            return -1;
        }
    }
    else
    {
        // printf("tengine cpp api : %s dims: n %d, h %d, w %d, c %d\n", __FUNCTION__, dims[0], dims[1], dims[2],
        // dims[3]); Mat m;
        if(dim_num == 4)
            t.create(dims[0], dims[3], dims[1], dims[2], 4, TENGINE_LAYOUT_NHWC);
        else
        {
            std::printf("Get tensor dim num is not 4, failed\n");
            return -1;
        }
    }

    int buffer_size = get_tensor_buffer_size(tensor);
    if (buffer_size != t.total())
    {
        std::printf("Get output tensor size failed\n");
        return -1;
    }
    void* buffer = (get_tensor_buffer(tensor));
    memcpy(t.data, buffer, buffer_size);

    return 0;
}

int Net::extract_tensor(const std::string& tensor_name, float*& buffer, int& buffer_size) const
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);

    tensor_t tensor = get_graph_tensor(graph, tensor_name.c_str());
    if(tensor == nullptr)
    {
        std::printf("Cannot find output tensor , tensor_name: %s \n", tensor_name.c_str());
        return -1;
    }

    buffer_size = get_tensor_buffer_size(tensor) / 4;
    buffer = ( float* )(get_tensor_buffer(tensor));

    return 0;
}

Tensor::Tensor(const Tensor& m)
{
    dim_num = m.dim_num;
    layout = m.layout;
    elem_size = m.elem_size;
    elem_num = m.elem_num;
    data = m.data;
    n = m.n;
    c = m.c;
    h = m.h;
    w = m.w;
}
Tensor::~Tensor()
{
    if(!data)
        free(data);
    dim_num = 0;
    layout = 0;
    elem_size = 0;
    elem_num = 0;
    data = nullptr;
    n = 0;
    c = 0;
    h = 0;
    w = 0;
}
void Tensor::fill(float _v) const
{
    int size = ( int )total();
    float* ptr = ( float* )data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif    // __ARM_NEON

#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if(nn > 0)
    {
        asm volatile("0:                             \n"
                     "subs       %w0, %w0, #1        \n"
                     "st1        {%4.4s}, [%1], #16  \n"
                     "bne        0b                  \n"
                     : "=r"(nn),    // %0
                       "=r"(ptr)    // %1
                     : "0"(nn), "1"(ptr),
                       "w"(_c)    // %4
                     : "cc", "memory");
    }
#else
    if(nn > 0)
    {
        asm volatile("0:                             \n"
                     "subs       %0, #1              \n"
                     "vst1.f32   {%4}, [%1]!         \n"
                     "bne        0b                  \n"
                     : "=r"(nn),    // %0
                       "=r"(ptr)    // %1
                     : "0"(nn), "1"(ptr),
                       "w"(_c)    // %4
                     : "cc", "memory");
    }
#endif    // __aarch64__
#endif    // __ARM_NEON
    for(; remain > 0; remain--)
    {
        *ptr++ = _v;
    }
}
template <typename T> void Tensor::fill(T _v)
{
    int size = total();
    T* ptr = ( T* )data;
    for(int i = 0; i < size; i++)
    {
        ptr[i] = _v;
    }
}

Tensor Tensor::clone() const
{
    if(empty())
        return Tensor();

    Tensor t;
    if(dim_num == 2)
        t.create(n, w, elem_size, layout);
    else if(dim_num == 3)
        t.create(n, h, w, elem_size, layout);
    else if(dim_num == 4)
        t.create(n, c, h, w, elem_size, layout);

    if(total() > 0)
    {
        memcpy(t.data, data, total() * elem_size);
    }

    return t;
}

void Tensor::create(int _n, int _w, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 2 && n == _n && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 2;
    layout = _layout;

    w = _w;
    h = 1;
    c = 1;
    n = _n;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}
void Tensor::create(int _n, int _h, int _w, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 3 && n == _n && h == _h && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 3;
    layout = _layout;

    w = _w;
    h = _h;
    c = 1;
    n = _n;
    elem_num = c * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}
void Tensor::create(int _n, int _c, int _h, int _w, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 4 && n == _n && c == _c && h == _h && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 4;
    layout = _layout;

    w = _w;
    h = _h;
    c = _c;
    n = _n;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}

bool Tensor::empty() const
{
    return data == nullptr || total() == 0;
}
size_t Tensor::total() const
{
    return elem_num * elem_size;
}

// // convenient construct from pixel data and resize to specific size, type is RGB/BGR
// static Tensor Tensor::from_pixels_resize(const unsigned char* pixels, int type, int c, int h, int w, int
// target_width,
//                                          int target_height)
// {
//     return
// }

// // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip, type maybe ONNX / MXNet /
// // Caffe
// void Tensor::substract_mean_normalize(const float* mean_vals, const float* norm_vals, const int type) {}

}    // namespace tengine
