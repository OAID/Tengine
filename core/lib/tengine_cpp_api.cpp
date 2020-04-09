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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#if __ARM_NEON
#include <arm_neon.h>
#endif
#include "tengine_cpp_api.h"

namespace tengine {
bool b_tengine_inited = false;

Net::Net()
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    net_lock_.lock();
    if(!b_tengine_inited)
    {
        init_tengine();
        b_tengine_inited = true;
    }
    net_lock_.unlock();
    graph = NULL;
    b_preruned = false;
}

Net::~Net()
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    postrun_graph(graph);
    destroy_graph(graph);
    net_lock_.lock();
    if(b_tengine_inited)
    {
        release_tengine();
        b_tengine_inited = false;
    }
    net_lock_.unlock();
    graph = NULL;
    b_preruned = false;
}

int Net::load_model(context_t context, const char* model_format, const char* model_file, ...)
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    graph = create_graph(NULL, model_format, model_file);

    if(graph == NULL)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    return 0;
}

int Net::input_shape(int n, int c, int h, int w, const char* tensor_name)
{
    // printf("tengine cpp api : %s name : %s %d:%d:%d:%d\n", __FUNCTION__, tensor_name, n, c, h, w);
    int dims[4];

    dims[0] = n;
    dims[1] = c;
    dims[2] = h;
    dims[3] = w;
    tensor_t tensor = get_graph_tensor(graph, tensor_name);
    if(tensor == NULL)
    {
        std::printf("Cannot find tensor name: %s\n", tensor_name);
        return -1;
    }

    int ret = set_tensor_shape(tensor, dims, 4);

    return ret;
}

int Net::input_tensor(const float* buffer, const int buffer_size, const char* tensor_name)
{
    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__, tensor_name);
    tensor_t tensor = get_graph_tensor(graph, tensor_name);
    if(tensor == NULL)
    {
        std::printf("Cannot find tensor name: %s\n", tensor_name);
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

int Net::prerun(void)
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);

    int ret = prerun_graph(graph);
    if(ret < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    return 0;
}

int Net::run(int block)
{
    if(!b_preruned)
    {
        if(prerun() == 0)
            b_preruned = true;
    }
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    int ret = run_graph(graph, block);

    return ret;
}

void Net::dump()
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);
    dump_graph(graph);
    return;
}

int Net::extract_tensor(float*& buffer, int& buffer_size, const char* tensor_name)
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);

    tensor_t tensor = get_graph_tensor(graph, tensor_name);
    if(tensor == NULL)
    {
        std::printf("Cannot find output tensor , tensor_name: %s \n", tensor_name);
        return -1;
    }

    buffer_size = get_tensor_buffer_size(tensor) / 4;
    buffer = ( float* )(get_tensor_buffer(tensor));

    return 0;
}

int Net::input_tensor(std::string name, Tensor& t)
{
    input_shape(t.n, t.c, t.h, t.w, name.c_str());
    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__,name.c_str());
    tensor_t tensor = get_graph_tensor(graph, name.c_str());
    if(tensor == NULL)
    {
        std::printf("Cannot find tensor name: %s\n", name.c_str());
        return -1;
    }
    int ret = set_tensor_buffer(tensor, ( void* )t.data, t.total() * t.elem_size);
    if(ret < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    return 0;
}

int Net::extract_tensor(std::string name, Tensor& t)
{
    // printf("tengine cpp api : %s\n", __FUNCTION__);

    tensor_t tensor = get_graph_tensor(graph, name.c_str());
    if(tensor == NULL)
    {
        std::printf("Cannot find output tensor , tensor_name: %s \n", name.c_str());
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
    // printf("tengine cpp api : %s dims: %d:%d:%d:%d\n", __FUNCTION__, dims[0], dims[1], dims[2], dims[3]);
    // Tensor m;
    if(dim_num == 4)
        t.create(dims[3], dims[2], dims[1], 4);
    else
    {
        /* code */
    }

    int buffer_size = get_tensor_buffer_size(tensor);
    void* buffer = (get_tensor_buffer(tensor));
    memcpy(t.data, buffer, buffer_size);

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
void Tensor::fill(float _v)
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
                     "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
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
    if(dim_num == 1)
        t.create(w, elem_size, layout);
    else if(dim_num == 2)
        t.create(w, h, elem_size, layout);
    else if(dim_num == 3)
        t.create(w, h, c, elem_size, layout);

    if(total() > 0)
    {
        memcpy(t.data, data, total() * elem_size);
    }

    return t;
}
Tensor Tensor::reshape(int _w) const
{
    if(w * h * c != _w)
        return Tensor();

    if(dim_num == 3)
    {
        Tensor m;
        m.create(_w, elem_size);
        size_t cstep = w * h;
        // flatten
        for(int i = 0; i < c; i++)
        {
            const void* ptr = ( unsigned char* )data + i * cstep * elem_size;
            void* mptr = ( unsigned char* )m.data + i * w * h * elem_size;
            memcpy(mptr, ptr, w * h * elem_size);
        }

        return m;
    }

    Tensor m = *this;

    m.dim_num = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;
    m.elem_num = w * h * c;

    return m;
}
Tensor Tensor::reshape(int _w, int _h) const
{
    if(w * h * c != _w * _h)
        return Tensor();

    if(dim_num == 3)
    {
        Tensor m;
        m.create(_w, _h, elem_size);
        size_t cstep = w * h;
        // flatten
        for(int i = 0; i < c; i++)
        {
            const void* ptr = ( unsigned char* )data + i * cstep * elem_size;
            void* mptr = ( unsigned char* )m.data + i * w * h * elem_size;
            memcpy(mptr, ptr, w * h * elem_size);
        }

        return m;
    }

    Tensor m = *this;

    m.dim_num = 1;
    m.w = _w;
    m.h = _h;
    m.c = 1;
    m.elem_num = w * h * c;

    return m;
}
Tensor Tensor::reshape(int _w, int _h, int _c) const
{
    if(w * h * c != _w * _h * _c)
        return Tensor();

    if(dim_num < 3)
    {
        if(_w * _h != w * h)
        {
            Tensor m;
            m.create(_w, _h, _c, elem_size);

            // align channel
            for(int i = 0; i < _c; i++)
            {
                const void* ptr = ( unsigned char* )data + i * _w * _h * elem_size;
                void* mptr = ( unsigned char* )m.data + i * m.w * m.h * m.elem_size;
                memcpy(mptr, ptr, _w * _h * elem_size);
            }

            return m;
        }
    }
    else if(c != _c)
    {
        // flatten and then align
        Tensor tmp = reshape(_w * _h * _c);
        return tmp.reshape(_w, _h, _c);
    }

    Tensor m = *this;

    m.dim_num = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.elem_num = w * h * c;

    return m;
}

void Tensor::create(int _w, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 1 && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 1;
    layout = _layout;

    w = _w;
    h = 1;
    c = 1;
    n = 1;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}
void Tensor::create(int _w, int _h, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 2 && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 2;
    layout = _layout;

    w = _w;
    h = _h;
    c = 1;
    n = 1;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}
void Tensor::create(int _w, int _h, int _c, size_t _elem_size, uint8_t _layout)
{
    if(dim_num == 3 && w == _w && elem_size == _elem_size && _layout == layout)
        return;

    elem_size = _elem_size;
    dim_num = 3;
    layout = _layout;

    w = _w;
    h = _h;
    c = _c;
    n = 1;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elem_size;
        data = malloc(totalsize);
    }
}

bool Tensor::empty() const
{
    return data == 0 || total() == 0;
}
size_t Tensor::total() const
{
    return elem_num;
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
