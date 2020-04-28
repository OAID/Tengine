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
 * https://github.com/Tencent/ncnn/blob/master/src/net.cpp
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
 * Copyright (c) 2020, OEPN AI LAB
 * Author: qtang@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <limits.h>
#include <math.h>
#include <algorithm>

#if __ARM_NEON
#include <arm_neon.h>
#endif
#include "net.h"
#include "cpu_device.h"

namespace ncnn {
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

int Net::load_param(const char* protopath)
{
    ncnn_param = protopath;
}

int Net::load_model(const char* modelpath)
{
    graph = create_graph(NULL, "ncnn", ncnn_param.c_str(), modelpath);

    if(graph == NULL)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    return 0;
}

int Net::set_kernel_mode(EKernelMode kernel_mode)
{
	const char* MODE_STR = "KERNEL_MODE";
    const char* PER_CHANNEL = "PER_CHANNEL";
    switch( kernel_mode)
    {
        case eKernelMode_Float32:
        {
            setenv(MODE_STR ,"1",1);
            unsetenv(PER_CHANNEL);
            break;
        }
        case eKernelMode_Int8:
        {
            setenv(MODE_STR ,"2",1);
            unsetenv(PER_CHANNEL);
            break;
        }
        case eKernelMode_Int8Perchannel:
        {
            setenv(MODE_STR ,"2",1);
            setenv("PER_CHANNEL" ,"1",1);
            break;
        }
    }

    return 0;
}

int Net::switch_wino(bool is_open)
{
	const char* WINO_STR = "NO_WINO";
	if( is_open )
	{
		unsetenv(WINO_STR);
	}
	else
	{
		setenv(WINO_STR,"1",1);
	}

	return 0;
}

int Net::set_worker_cpu_list(const int* cpu_list,int num)
{
	set_working_cpu(cpu_list,num);
    return 0;
}

int Net::set_device(std::string device)
{
    if(!device.empty())
    {
        set_graph_device(graph, device.c_str());
    }
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

int Net::input(std::string name, Mat& t)
{
    input_shape(t.n, t.c, t.h, t.w, name.c_str());
    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__,name.c_str());
    tensor_t tensor = get_graph_tensor(graph, name.c_str());
    if(tensor == NULL)
    {
        std::printf("Cannot find tensor name: %s\n", name.c_str());
        return -1;
    }
    int ret = set_tensor_buffer(tensor, ( void* )t.data, t.total() * t.elemsize);
    if(ret < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    return 0;
}

int Net::input_tensor(int node_index, int tensor_index, Mat& t)
{
    int dims[4];

    dims[0] = t.n;
    dims[1] = t.c;
    dims[2] = t.h;
    dims[3] = t.w;

    // printf("tengine cpp api : %s tensor_name:%s\n", __FUNCTION__,name.c_str());
    tensor_t tensor = get_graph_input_tensor(graph, node_index, tensor_index);
    if(tensor == NULL)
    {
        std::printf("Cannot find tensor node_index: %d, tensor_index: %d\n", node_index, tensor_index);
        return -1;
    }
    int ret = set_tensor_buffer(tensor, ( void* )t.data, t.total() * t.elemsize);
    if(ret < 0)
    {
        std::printf("Set buffer for tensor failed\n");
        return -1;
    }

    set_tensor_shape(tensor, dims, 4);

    return 0;
}

int Net::extract(std::string name, Mat& t)
{
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

    int layout = get_tensor_layout(tensor);
    if (layout == TENGINE_LAYOUT_NCHW)
    {
        // printf("tengine cpp api : %s dims: n %d, c %d, h %d, w %d\n", __FUNCTION__, dims[0], dims[1], dims[2], dims[3]);
        // Mat m;
        if(dim_num == 4)
            t.create(dims[3], dims[2], dims[1], 4);
        else
        {
            std::printf("Get tensor dim num is not 4, failed\n");
            return -1;
        }
    }
    else
    {
        // printf("tengine cpp api : %s dims: n %d, h %d, w %d, c %d\n", __FUNCTION__, dims[0], dims[1], dims[2], dims[3]);
        // Mat m;
        if(dim_num == 4)
            t.create(dims[2], dims[1], dims[3], 4);
        else
        {
            std::printf("Get tensor dim num is not 4, failed\n");
            return -1;
        }        
    }

    int buffer_size = get_tensor_buffer_size(tensor);
    void* buffer = (get_tensor_buffer(tensor));
    memcpy(t.data, buffer, buffer_size);

    return 0;
}

int Net::extract_tensor(int node_index, int tensor_index, Mat& t)
{
    tensor_t tensor = get_graph_output_tensor(graph, node_index, tensor_index);

    if(tensor == NULL)
    {
        std::printf("Cannot find output tensor, node_index: %d, tensor_index: %d\n", node_index, tensor_index);
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
    // Mat m;
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


Mat::Mat(const Mat& m)
{
    dim_num = m.dim_num;
    layout = m.layout;
    elemsize = m.elemsize;
    elem_num = m.elem_num;
    data = m.data;
    n = m.n;
    c = m.c;
    h = m.h;
    w = m.w;
}

Mat::~Mat()
{
    if(!data)
        free(data);
    dim_num = 0;
    layout = 0;
    elemsize = 0;
    elem_num = 0;
    data = nullptr;
    n = 0;
    c = 0;
    h = 0;
    w = 0;
}

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    int size = w * h;

    if (mean_vals && !norm_vals)
    {
        // substract mean only
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = channel(q);//data + cstep * q;
            const float mean = mean_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean)     // %4
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vsub.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean)     // %4
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr -= mean;
                ptr++;
            }
        }
    }
    else if (!mean_vals && norm_vals)
    {
        // normalize only
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = channel(q);//data + cstep * q;
            const float norm = norm_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(norm)     // %4
                : "cc", "memory", "v0", "v1"
            );
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmul.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(norm)     // %4
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr *= norm;
                ptr++;
            }
        }
    }
    else if (mean_vals && norm_vals)
    {
        // substract mean and normalize
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = channel(q);//data + cstep * q;
            const float mean = mean_vals[q];
            const float norm = norm_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "dup        v2.4s, %w5            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "fmul       v0.4s, v0.4s, v2.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean),    // %4
                  "r"(norm)     // %5
                : "cc", "memory", "v0", "v1", "v2"
            );  
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "vdup.f32   q2, %5              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vsub.f32   q0, q0, q1          \n"
                "vmul.f32   q0, q0, q2          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean),    // %4
                  "r"(norm)     // %5
                : "cc", "memory", "q0", "q1", "q2"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr = (*ptr - mean) * norm;
                ptr++;
            }
        }
    }
}

void Mat::fill(float _v)
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
template <typename T> void Mat::fill(T _v)
{
    int size = total();
    T* ptr = ( T* )data;
    for(int i = 0; i < size; i++)
    {
        ptr[i] = _v;
    }
}

Mat Mat::clone() const
{
    if(empty())
        return Mat();

    Mat t;
    if(dim_num == 1)
        t.create(w, elemsize, layout);
    else if(dim_num == 2)
        t.create(w, h, elemsize, layout);
    else if(dim_num == 3)
        t.create(w, h, c, elemsize, layout);

    if(total() > 0)
    {
        memcpy(t.data, data, total() * elemsize);
    }

    return t;
}
Mat Mat::reshape(int _w) const
{
    if(w * h * c != _w)
        return Mat();

    if(dim_num == 3)
    {
        Mat m;
        m.create(_w, elemsize);
        size_t cstep = w * h;
        // flatten
        for(int i = 0; i < c; i++)
        {
            const void* ptr = ( unsigned char* )data + i * cstep * elemsize;
            void* mptr = ( unsigned char* )m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dim_num = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;
    m.elem_num = w * h * c;

    return m;
}
Mat Mat::reshape(int _w, int _h) const
{
    if(w * h * c != _w * _h)
        return Mat();

    if(dim_num == 3)
    {
        Mat m;
        m.create(_w, _h, elemsize);
        size_t cstep = w * h;
        // flatten
        for(int i = 0; i < c; i++)
        {
            const void* ptr = ( unsigned char* )data + i * cstep * elemsize;
            void* mptr = ( unsigned char* )m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dim_num = 1;
    m.w = _w;
    m.h = _h;
    m.c = 1;
    m.elem_num = w * h * c;

    return m;
}
Mat Mat::reshape(int _w, int _h, int _c) const
{
    if(w * h * c != _w * _h * _c)
        return Mat();

    if(dim_num < 3)
    {
        if(_w * _h != w * h)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize);

            // align channel
            for(int i = 0; i < _c; i++)
            {
                const void* ptr = ( unsigned char* )data + i * _w * _h * elemsize;
                void* mptr = ( unsigned char* )m.data + i * m.w * m.h * m.elemsize;
                memcpy(mptr, ptr, _w * _h * elemsize);
            }

            return m;
        }
    }
    else if(c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c);
        return tmp.reshape(_w, _h, _c);
    }

    Mat m = *this;

    m.dim_num = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.elem_num = w * h * c;

    return m;
}

void Mat::create(int _w, size_t _elemsize, uint8_t _layout)
{
    if(dim_num == 1 && w == _w && elemsize == _elemsize && _layout == layout)
        return;

    elemsize = _elemsize;
    dim_num = 1;
    layout = _layout;

    w = _w;
    h = 1;
    c = 1;
    n = 1;
    cstep = w;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = malloc(totalsize);
    }
}
void Mat::create(int _w, int _h, size_t _elemsize, uint8_t _layout)
{
    if(dim_num == 2 && w == _w && elemsize == _elemsize && _layout == layout)
        return;

    elemsize = _elemsize;
    dim_num = 2;
    layout = _layout;

    w = _w;
    h = _h;
    c = 1;
    n = 1;
    cstep = w * h;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = malloc(totalsize);
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, uint8_t _layout)
{
    if(w == _w && elemsize == _elemsize && _layout == layout)
        return;

    w = _w;
    h = _h;
    c = _c;
    n = 1;
    cstep = w * h;
    elemsize = _elemsize;
    dim_num = 4;
    layout = _layout;
    elem_num = w * h * c * n;

    if(total() > 0)
    {
        size_t totalsize = total() * elemsize;
        data = malloc(totalsize);
    }
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

size_t Mat::total() const
{
    return elem_num;
}

inline Mat Mat::shape() const
{
    if (dims == 1)
        return Mat(w, (void*)0);
    if (dims == 2)
        return Mat(w, h, (void*)0);
    if (dims == 3)
        return Mat(w, h, c, (void*)0);

    return Mat();
}

// inline Mat Mat::channel(int _c)
// {
//     return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize);
// }

// inline const Mat Mat::channel(int _c) const
// {
//     return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize);
// }

// inline float* Mat::row(int y)
// {
//     return (float*)((unsigned char*)data + w * y * elemsize);
// }

// inline const float* Mat::row(int y) const
// {
//     return (const float*)((unsigned char*)data + w * y * elemsize);
// }

// template <typename T>
// inline T* Mat::row(int y)
// {
//     return (T*)((unsigned char*)data + w * y * elemsize);
// }

// template <typename T>
// inline const T* Mat::row(int y) const
// {
//     return (const T*)((unsigned char*)data + w * y * elemsize);
// }

// inline Mat Mat::channel_range(int _c, int channels)
// {
//     return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize);
// }

// inline const Mat Mat::channel_range(int _c, int channels) const
// {
//     return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize);
// }

// inline Mat Mat::row_range(int y, int rows)
// {
//     return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize);
// }

// inline const Mat Mat::row_range(int y, int rows) const
// {
//     return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize);
// }

// inline Mat Mat::range(int x, int n)
// {
//     return Mat(n, (unsigned char*)data + x * elemsize, elemsize);
// }

// inline const Mat Mat::range(int x, int n) const
// {
//     return Mat(n, (unsigned char*)data + x * elemsize, elemsize);
// }

// template <typename T>
// inline Mat::operator T*()
// {
//     return (T*)data;
// }

// template <typename T>
// inline Mat::operator const T*() const
// {
//     return (const T*)data;
// }

// inline float& Mat::operator[](size_t i)
// {
//     return ((float*)data)[i];
// }

// inline const float& Mat::operator[](size_t i) const
// {
//     return ((const float*)data)[i];
// }

// from pixel resize

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0]*a0 + S1p[1]*a1) >> 4;

                ialphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcstride * (sy);
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0]*a0 + S0p[1]*a1) >> 4;
                rows1p[dx] = (S1p[0]*a0 + S1p[1]*a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.s16   d16, %8         \n"
            "mov        r4, #2          \n"
            "vdup.s16   d17, %9         \n"
            "vdup.s32   q12, r4         \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "0:                         \n"
            "vmull.s16  q0, d2, d16     \n"
            "vmull.s16  q1, d3, d16     \n"
            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vmull.s16  q2, d6, d17     \n"
            "vmull.s16  q3, d7, d17     \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%2]!    \n"
            "subs       %3, #1          \n"
            "bne        0b              \n"
            "sub        %0, #16         \n"
            "sub        %1, #16         \n"
            : "=r"(rows0p), // %0
              "=r"(rows1p), // %1
              "=r"(Dp),     // %2
              "=r"(nn)      // %3
            : "0"(rows0p),
              "1"(rows1p),
              "2"(Dp),
              "3"(nn),
              "r"(b0),      // %8
              "r"(b1)       // %9
            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx*2;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w*2+2, (size_t)2u);
    Mat rowsbuf1(w*2+2, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0a1XX = vld1_s16(ialphap);
                int16x4_t _a0a0a1a1 = vzip_s16(_a0a1XX, _a0a1XX).val[0];
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p+1, _S1, 1);
                _S1 = vld1_lane_u8(S1p+2, _S1, 2);
                _S1 = vld1_lane_u8(S1p+3, _S1, 3);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x4_t _S1ma0a1 = vmull_s16(_S1lowhigh, _a0a0a1a1);
                int32x2_t _rows1low = vadd_s32(vget_low_s32(_S1ma0a1), vget_high_s32(_S1ma0a1));
                int32x4_t _rows1 = vcombine_s32(_rows1low, vget_high_s32(_S1ma0a1));
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                rows1p[0] = (S1p[0]*a0 + S1p[2]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[3]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcstride * (sy);
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p+1, _S0, 1);
                _S0 = vld1_lane_u8(S0p+2, _S0, 2);
                _S0 = vld1_lane_u8(S0p+3, _S0, 3);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p+1, _S1, 1);
                _S1 = vld1_lane_u8(S1p+2, _S1, 2);
                _S1 = vld1_lane_u8(S1p+3, _S1, 3);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0lowhigh = vget_low_s16(_S016);
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x2x2_t _S0S1low_S0S1high = vtrn_s32(vreinterpret_s32_s16(_S0lowhigh), vreinterpret_s32_s16(_S1lowhigh));
                int32x4_t _rows01 = vmull_s16(vreinterpret_s16_s32(_S0S1low_S0S1high.val[0]), _a0);
                _rows01 = vmlal_s16(_rows01, vreinterpret_s16_s32(_S0S1low_S0S1high.val[1]), _a1);
                int16x4_t _rows01_sr4 = vshrn_n_s32(_rows01, 4);
                int16x4_t _rows1_sr4 = vext_s16(_rows01_sr4, _rows01_sr4, 2);
                vst1_s16(rows0p, _rows01_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0]*a0 + S0p[2]*a1) >> 4;
                rows0p[1] = (S0p[1]*a0 + S0p[3]*a1) >> 4;
                rows1p[0] = (S1p[0]*a0 + S1p[2]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[3]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 2;
                rows1p += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 2) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 2) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.s16   d16, %8         \n"
            "mov        r4, #2          \n"
            "vdup.s16   d17, %9         \n"
            "vdup.s32   q12, r4         \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "0:                         \n"
            "vmull.s16  q0, d2, d16     \n"
            "vmull.s16  q1, d3, d16     \n"
            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vmull.s16  q2, d6, d17     \n"
            "vmull.s16  q3, d7, d17     \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%2]!    \n"
            "subs       %3, #1          \n"
            "bne        0b              \n"
            "sub        %0, #16         \n"
            "sub        %1, #16         \n"
            : "=r"(rows0p), // %0
              "=r"(rows1p), // %1
              "=r"(Dp),     // %2
              "=r"(nn)      // %3
            : "0"(rows0p),
              "1"(rows1p),
              "2"(Dp),
              "3"(nn),
              "r"(b0),      // %8
              "r"(b1)       // %9
            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx*3;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w*3+1, (size_t)2u);
    Mat rowsbuf1(w*3+1, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p+1, _S1, 1);
                _S1 = vld1_lane_u8(S1p+2, _S1, 2);
                _S1 = vld1_lane_u8(S1p+3, _S1, 3);
                _S1 = vld1_lane_u8(S1p+4, _S1, 4);
                _S1 = vld1_lane_u8(S1p+5, _S1, 5);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0]*a0 + S1p[3]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[4]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[5]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 3;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcstride * (sy);
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p+1, _S0, 1);
                _S0 = vld1_lane_u8(S0p+2, _S0, 2);
                _S0 = vld1_lane_u8(S0p+3, _S0, 3);
                _S0 = vld1_lane_u8(S0p+4, _S0, 4);
                _S0 = vld1_lane_u8(S0p+5, _S0, 5);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p+1, _S1, 1);
                _S1 = vld1_lane_u8(S1p+2, _S1, 2);
                _S1 = vld1_lane_u8(S1p+3, _S1, 3);
                _S1 = vld1_lane_u8(S1p+4, _S1, 4);
                _S1 = vld1_lane_u8(S1p+5, _S1, 5);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0]*a0 + S0p[3]*a1) >> 4;
                rows0p[1] = (S0p[1]*a0 + S0p[4]*a1) >> 4;
                rows0p[2] = (S0p[2]*a0 + S0p[5]*a1) >> 4;
                rows1p[0] = (S1p[0]*a0 + S1p[3]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[4]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[5]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 3) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 3) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.s16   d16, %8         \n"
            "mov        r4, #2          \n"
            "vdup.s16   d17, %9         \n"
            "vdup.s32   q12, r4         \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "0:                         \n"
            "vmull.s16  q0, d2, d16     \n"
            "vmull.s16  q1, d3, d16     \n"
            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vmull.s16  q2, d6, d17     \n"
            "vmull.s16  q3, d7, d17     \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%2]!    \n"
            "subs       %3, #1          \n"
            "bne        0b              \n"
            "sub        %0, #16         \n"
            "sub        %1, #16         \n"
            : "=r"(rows0p), // %0
              "=r"(rows1p), // %1
              "=r"(Dp),     // %2
              "=r"(nn)      // %3
            : "0"(rows0p),
              "1"(rows1p),
              "2"(Dp),
              "3"(nn),
              "r"(b0),      // %8
              "r"(b1)       // %9
            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx*4;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w*4, (size_t)2u);
    Mat rowsbuf1(w*4, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 4)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0]*a0 + S1p[4]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[5]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[6]*a1) >> 4;
                rows1p[3] = (S1p[3]*a0 + S1p[7]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 4;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcstride * (sy);
            const unsigned char *S1 = src + srcstride * (sy+1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = vld1_u8(S0p);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vget_high_s16(_S016);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0]*a0 + S0p[4]*a1) >> 4;
                rows0p[1] = (S0p[1]*a0 + S0p[5]*a1) >> 4;
                rows0p[2] = (S0p[2]*a0 + S0p[6]*a1) >> 4;
                rows0p[3] = (S0p[3]*a0 + S0p[7]*a1) >> 4;
                rows1p[0] = (S1p[0]*a0 + S1p[4]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[5]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[6]*a1) >> 4;
                rows1p[3] = (S1p[3]*a0 + S1p[7]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

#if __ARM_NEON
        int nn = (w * 4) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 4) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.s16   d16, %8         \n"
            "mov        r4, #2          \n"
            "vdup.s16   d17, %9         \n"
            "vdup.s32   q12, r4         \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "0:                         \n"
            "vmull.s16  q0, d2, d16     \n"
            "vmull.s16  q1, d3, d16     \n"
            "vorr.s32   q10, q12, q12   \n"
            "vorr.s32   q11, q12, q12   \n"
            "vmull.s16  q2, d6, d17     \n"
            "vmull.s16  q3, d7, d17     \n"
            "vsra.s32   q10, q0, #16    \n"
            "vsra.s32   q11, q1, #16    \n"
            "pld        [%0, #128]      \n"
            "vld1.s16   {d2-d3}, [%0 :128]!\n"
            "vsra.s32   q10, q2, #16    \n"
            "vsra.s32   q11, q3, #16    \n"
            "pld        [%1, #128]      \n"
            "vld1.s16   {d6-d7}, [%1 :128]!\n"
            "vshrn.s32  d20, q10, #2    \n"
            "vshrn.s32  d21, q11, #2    \n"
            "vqmovun.s16 d20, q10        \n"
            "vst1.8     {d20}, [%2]!    \n"
            "subs       %3, #1          \n"
            "bne        0b              \n"
            "sub        %0, #16         \n"
            "sub        %1, #16         \n"
            : "=r"(rows0p), // %0
              "=r"(rows1p), // %1
              "=r"(Dp),     // %2
              "=r"(nn)      // %3
            : "0"(rows0p),
              "1"(rows1p),
              "2"(Dp),
              "3"(nn),
              "r"(b0),      // %8
              "r"(b1)       // %9
            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w);
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2);
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3);
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4);
}

void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    resize_bilinear_c1(srcY, srcw, srch, dstY, w, h);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    resize_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2);
}

// from pixel
static int from_rgb(const unsigned char* rgb, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 3, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2+4, _bhigh);

            rgb += 3*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgb),    // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2)    // %4
            : "0"(nn),
              "1"(rgb),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x3_t _rgb;
            _rgb.val[0] = vqmovun_s16(_r16);
            _rgb.val[1] = vqmovun_s16(_g16);
            _rgb.val[2] = vqmovun_s16(_b16);

            vst3_u8(rgb, _rgb);

            rgb += 3*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            rgb[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[2] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_gray(const unsigned char* gray, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 1, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 4;
        int remain = w - (nn << 4);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _gray = vld1q_u8(gray);
            uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
            uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

            float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
            float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
            float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
            float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

            vst1q_f32(ptr, _graylow_0);
            vst1q_f32(ptr+4, _grayhigh_0);
            vst1q_f32(ptr+8, _graylow_1);
            vst1q_f32(ptr+12, _grayhigh_1);

            gray += 16;
            ptr += 16;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.u8    {d0,d1}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vst1.f32   {d4-d7}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(gray),   // %1
              "=r"(ptr)     // %2
            : "0"(nn),
              "1"(gray),
              "2"(ptr)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = *gray;

            gray++;
            ptr++;
        }

        gray += wgap;
    }

    return 0;
}

static void to_gray(const Mat& m, unsigned char* gray, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _glow = vld1q_f32(ptr);
            float32x4_t _ghigh = vld1q_f32(ptr+4);

            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));

            uint8x8_t _gray = vqmovun_s16(_g16);

            vst1_u8(gray, _gray);

            gray += 8;
            ptr += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *gray = SATURATE_CAST_UCHAR(*ptr);

            gray++;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        gray += wgap;
    }
}

static int from_rgba(const unsigned char* rgba, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 4, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));
            int16x8_t _a16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[3]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));
            float32x4_t _alow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_a16)));
            float32x4_t _ahigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_a16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2+4, _bhigh);
            vst1q_f32(ptr3, _alow);
            vst1q_f32(ptr3+4, _ahigh);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u8   q11, d3             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vmovl.u16  q10, d22            \n"
            "vmovl.u16  q11, d23            \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "vcvt.f32.u32   q9, q9          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q10, q10        \n"
            "vcvt.f32.u32   q11, q11        \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "vst1.f32   {d20-d23}, [%5 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgba),   // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2),   // %4
              "=r"(ptr3)    // %5
            : "0"(nn),
              "1"(rgba),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2),
              "5"(ptr3)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2+4);
            float32x4_t _alow = vld1q_f32(ptr3);
            float32x4_t _ahigh = vld1q_f32(ptr3+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));
            int16x8_t _a16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_alow)), vmovn_s32(vcvtq_s32_f32(_ahigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = vqmovun_s16(_a16);

            vst4_u8(rgba, _rgba);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = SATURATE_CAST_UCHAR(*ptr3);

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgb2bgr(const unsigned char* rgb, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 3, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0+4, _bhigh);

            rgb += 3*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%2 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgb),    // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2)    // %4
            : "0"(nn),
              "1"(rgb),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgb[2];
            *ptr1 = rgb[1];
            *ptr2 = rgb[0];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_bgr2rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr2);
            float32x4_t _rhigh = vld1q_f32(ptr2+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr0);
            float32x4_t _bhigh = vld1q_f32(ptr0+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x3_t _rgb;
            _rgb.val[0] = vqmovun_s16(_r16);
            _rgb.val[1] = vqmovun_s16(_g16);
            _rgb.val[2] = vqmovun_s16(_b16);

            vst3_u8(rgb, _rgb);

            rgb += 3*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            rgb[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[0] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_rgb2gray(const unsigned char* rgb, int w, int h, int stride, Mat& m)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn>0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);

            uint16x8_t _y16 = vmull_u8(_rgb.val[0], _R2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr+4, _yhigh);

            rgb += 3*8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmull.u8   q2, d0, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d2, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgb),    // %1
              "=r"(ptr)     // %2
            : "0"(nn),
              "1"(rgb),
              "2"(ptr),
              "r"(R2Y),     // %6
              "r"(G2Y),     // %7
              "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = static_cast<float>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);

            rgb += 3;
            ptr++;
        }

        rgb += wgap;
    }

    return 0;
}

static int from_rgb2rgba(const unsigned char* rgb, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 4, 4u);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb(rgb, w, h, stride, rgb_channels);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_rgb2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_bgr2gray(const unsigned char* bgr, int w, int h, int stride, Mat& m)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn>0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(bgr);

            uint16x8_t _y16 = vmull_u8(_rgb.val[2], _R2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[0], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr+4, _yhigh);

            bgr += 3*8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmull.u8   q2, d2, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d0, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(bgr),    // %1
              "=r"(ptr)     // %2
            : "0"(nn),
              "1"(bgr),
              "2"(ptr),
              "r"(R2Y),     // %6
              "r"(G2Y),     // %7
              "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = static_cast<float>((bgr[2] * R2Y + bgr[1] * G2Y + bgr[0] * B2Y) >> Y_shift);

            bgr += 3;
            ptr++;
        }

        bgr += wgap;
    }

    return 0;
}

static int from_bgr2rgba(const unsigned char* bgr, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 4, 4u);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb2bgr(bgr, w, h, stride, rgb_channels);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_bgr2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr2);
            float32x4_t _rhigh = vld1q_f32(ptr2+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr0);
            float32x4_t _bhigh = vld1q_f32(ptr0+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_gray2rgb(const unsigned char* gray, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 3, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 4;
        int remain = w - (nn << 4);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x16_t _gray = vld1q_u8(gray);
            uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
            uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

            float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
            float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
            float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
            float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

            vst1q_f32(ptr0, _graylow_0);
            vst1q_f32(ptr0+4, _grayhigh_0);
            vst1q_f32(ptr0+8, _graylow_1);
            vst1q_f32(ptr0+12, _grayhigh_1);

            vst1q_f32(ptr1, _graylow_0);
            vst1q_f32(ptr1+4, _grayhigh_0);
            vst1q_f32(ptr1+8, _graylow_1);
            vst1q_f32(ptr1+12, _grayhigh_1);

            vst1q_f32(ptr2, _graylow_0);
            vst1q_f32(ptr2+4, _grayhigh_0);
            vst1q_f32(ptr2+8, _graylow_1);
            vst1q_f32(ptr2+12, _grayhigh_1);

            gray += 16;
            ptr0 += 16;
            ptr1 += 16;
            ptr2 += 16;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.u8    {d0,d1}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vst1.f32   {d4-d7}, [%2 :128]! \n"
            "vst1.f32   {d0-d3}, [%3 :128]! \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vst1.f32   {d4-d7}, [%4 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(gray),   // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2)    // %4
            : "0"(nn),
              "1"(gray),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = *gray;
            *ptr1 = *gray;
            *ptr2 = *gray;

            gray++;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        gray += wgap;
    }

    return 0;
}

static int from_gray2rgba(const unsigned char* gray, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 4, 4u);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_gray2rgb(gray, w, h, stride, rgb_channels);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_gray2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn>0; nn--)
        {
            float32x4_t _glow = vld1q_f32(ptr);
            float32x4_t _ghigh = vld1q_f32(ptr+4);

            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));

            uint8x8_t _gray = vqmovun_s16(_g16);

            uint8x8x4_t _rgba;
            _rgba.val[0] = _gray;
            _rgba.val[1] = _gray;
            _rgba.val[2] = _gray;
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4*8;
            ptr += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            unsigned char gray = SATURATE_CAST_UCHAR(*ptr);
            rgba[0] = gray;
            rgba[1] = gray;
            rgba[2] = gray;
            rgba[3] = 255;

            rgba += 4;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgba2rgb(const unsigned char* rgba, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 3, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2+4, _bhigh);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgba),   // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2)    // %4
            : "0"(nn),
              "1"(rgba),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgr(const unsigned char* rgba, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 3, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0+4, _bhigh);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%2 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgba),   // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2)    // %4
            : "0"(nn),
              "1"(rgba),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2gray(const unsigned char* rgba, int w, int h, int stride, Mat& m)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn>0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);

            uint16x8_t _y16 = vmull_u8(_rgba.val[0], _R2Y);
            _y16 = vmlal_u8(_y16, _rgba.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgba.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr+4, _yhigh);

            rgba += 4*8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmull.u8   q2, d0, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d2, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgba),   // %1
              "=r"(ptr)     // %2
            : "0"(nn),
              "1"(rgba),
              "2"(ptr),
              "r"(R2Y),     // %6
              "r"(G2Y),     // %7
              "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = static_cast<float>((rgba[0] * R2Y + rgba[1] * G2Y + rgba[2] * B2Y) >> Y_shift);

            rgba += 4;
            ptr++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgra(const unsigned char* rgba, int w, int h, int stride, Mat& m)
{
    m.create(w, h, 4, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));
            int16x8_t _a16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[3]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));
            float32x4_t _alow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_a16)));
            float32x4_t _ahigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_a16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2+4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1+4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0+4, _bhigh);
            vst1q_f32(ptr3, _alow);
            vst1q_f32(ptr3+4, _ahigh);

            rgba += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u8   q11, d3             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vmovl.u16  q10, d22            \n"
            "vmovl.u16  q11, d23            \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vcvt.f32.u32   q10, q10        \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vcvt.f32.u32   q11, q11        \n"
            "vst1.f32   {d16-d19}, [%2 :128]!\n"
            "vst1.f32   {d20-d23}, [%5 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(rgba),   // %1
              "=r"(ptr0),   // %2
              "=r"(ptr1),   // %3
              "=r"(ptr2),   // %4
              "=r"(ptr3)    // %5
            : "0"(nn),
              "1"(rgba),
              "2"(ptr0),
              "3"(ptr1),
              "4"(ptr2),
              "5"(ptr3)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba2bgra(const Mat& m, unsigned char* bgra, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y=0; y<h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn>0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0+4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1+4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2+4);
            float32x4_t _alow = vld1q_f32(ptr3);
            float32x4_t _ahigh = vld1q_f32(ptr3+4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));
            int16x8_t _a16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_alow)), vmovn_s32(vcvtq_s32_f32(_ahigh)));

            uint8x8x4_t _bgra;
            _bgra.val[0] = vqmovun_s16(_b16);
            _bgra.val[1] = vqmovun_s16(_g16);
            _bgra.val[2] = vqmovun_s16(_r16);
            _bgra.val[3] = vqmovun_s16(_a16);

            vst4_u8(bgra, _bgra);

            bgra += 4*8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            bgra[0] = SATURATE_CAST_UCHAR(*ptr2);
            bgra[1] = SATURATE_CAST_UCHAR(*ptr1);
            bgra[2] = SATURATE_CAST_UCHAR(*ptr0);
            bgra[3] = SATURATE_CAST_UCHAR(*ptr3);

            bgra += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        bgra += wgap;
    }
}

static int from_bgra2gray(const unsigned char* bgra, int w, int h, int stride, Mat& m)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y=0; y<h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn>0; nn--)
        {
            uint8x8x4_t _bgra = vld4_u8(bgra);

            uint16x8_t _y16 = vmull_u8(_bgra.val[2], _R2Y);
            _y16 = vmlal_u8(_y16, _bgra.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _bgra.val[0], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr+4, _yhigh);

            bgra += 4*8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmull.u8   q2, d2, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d0, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(bgra),   // %1
              "=r"(ptr)     // %2
            : "0"(nn),
              "1"(bgra),
              "2"(ptr),
              "r"(R2Y),     // %6
              "r"(G2Y),     // %7
              "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = static_cast<float>((bgra[2] * R2Y + bgra[1] * G2Y + bgra[0] * B2Y) >> Y_shift);

            bgra += 4;
            ptr++;
        }

        bgra += wgap;
    }

    return 0;
}

void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* vuptr = yuv420sp + w * h;

#if __ARM_NEON
    uint8x8_t _v128 = vdup_n_u8(128);
    int8x8_t _v90 = vdup_n_s8(90);
    int8x8_t _v46 = vdup_n_s8(46);
    int8x8_t _v22 = vdup_n_s8(22);
    int8x8_t _v113 = vdup_n_s8(113);
#endif // __ARM_NEON

    for (int y=0; y<h; y+=2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w*3;

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn>0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _vvuu = vreinterpret_s8_u8(vsub_u8(vld1_u8(vuptr), _v128));
            int8x8x2_t _vvvvuuuu = vtrn_s8(_vvuu, _vvuu);
            int8x8_t _vv = _vvvvuuuu.val[0];
            int8x8_t _uu = _vvvvuuuu.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _v90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _v46);
            _g0 = vmlsl_s8(_g0, _uu, _v22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _v113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _v90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _v46);
            _g1 = vmlsl_s8(_g1, _uu, _v22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _v113);

            uint8x8x3_t _rgb0;
            _rgb0.val[0] = vqshrun_n_s16(_r0, 6);
            _rgb0.val[1] = vqshrun_n_s16(_g0, 6);
            _rgb0.val[2] = vqshrun_n_s16(_b0, 6);

            uint8x8x3_t _rgb1;
            _rgb1.val[0] = vqshrun_n_s16(_r1, 6);
            _rgb1.val[1] = vqshrun_n_s16(_g1, 6);
            _rgb1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(rgb0, _rgb0);
            vst3_u8(rgb1, _rgb1);

            yptr0 += 8;
            yptr1 += 8;
            vuptr += 8;
            rgb0 += 24;
            rgb1 += 24;
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "pld        [%3, #128]          \n"
            "vld1.u8    {d2}, [%3]!         \n"
            "vsub.s8    d2, d2, %12         \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.u8    {d0}, [%1]!         \n"
            "pld        [%2, #128]          \n"
            "vld1.u8    {d1}, [%2]!         \n"
            "vshll.u8   q2, d0, #6          \n"
            "vorr       d3, d2, d2          \n"
            "vshll.u8   q3, d1, #6          \n"
            "vorr       q9, q2, q2          \n"
            "vtrn.s8    d2, d3              \n"
            "vorr       q11, q3, q3         \n"
            "vmlsl.s8   q9, d2, %14         \n"
            "vorr       q8, q2, q2          \n"
            "vmlsl.s8   q11, d2, %14        \n"
            "vorr       q10, q3, q3         \n"
            "vmlal.s8   q8, d2, %13         \n"
            "vmlal.s8   q2, d3, %16         \n"
            "vmlal.s8   q10, d2, %13        \n"
            "vmlsl.s8   q9, d3, %15         \n"
            "vmlal.s8   q3, d3, %16         \n"
            "vmlsl.s8   q11, d3, %15        \n"
            "vqshrun.s16 d24, q8, #6        \n"
            "vqshrun.s16 d26, q2, #6        \n"
            "vqshrun.s16 d4, q10, #6        \n"
            "vqshrun.s16 d25, q9, #6        \n"
            "vqshrun.s16 d6, q3, #6         \n"
            "vqshrun.s16 d5, q11, #6        \n"
            "pld        [%3, #128]          \n"
            "vld1.u8    {d2}, [%3]!         \n"
            "subs       %0, #1              \n"
            "vst3.u8    {d24-d26}, [%4]!    \n"
            "vsub.s8    d2, d2, %12         \n"
            "vst3.u8    {d4-d6}, [%5]!      \n"
            "bne        0b                  \n"
            "sub        %3, #8              \n"
            : "=r"(nn),     // %0
              "=r"(yptr0),  // %1
              "=r"(yptr1),  // %2
              "=r"(vuptr),  // %3
              "=r"(rgb0),   // %4
              "=r"(rgb1)    // %5
            : "0"(nn),
              "1"(yptr0),
              "2"(yptr1),
              "3"(vuptr),
              "4"(rgb0),
              "5"(rgb1),
              "w"(_v128),   // %12
              "w"(_v90),    // %13
              "w"(_v46),    // %14
              "w"(_v22),    // %15
              "w"(_v113)    // %16
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "d26"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain>0; remain-=2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int v = vuptr[0] - 128;
            int u = vuptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            vuptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2*w;
        rgb += 2*3*w;
    }
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 3);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 1);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 4);
    }

    // unknown convert type
    return Mat();
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, int stride)
{
    Mat m;

    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            from_rgb2bgr(pixels, w, h, stride, m);
            break;
        case PIXEL_RGB2GRAY:
            from_rgb2gray(pixels, w, h, stride, m);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            from_rgb2rgba(pixels, w, h, stride, m);
            break;
        case PIXEL_BGR2GRAY:
            from_bgr2gray(pixels, w, h, stride, m);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            from_bgr2rgba(pixels, w, h, stride, m);
            break;
        case PIXEL_GRAY2RGB:
        case PIXEL_GRAY2BGR:
            from_gray2rgb(pixels, w, h, stride, m);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            from_gray2rgba(pixels, w, h, stride, m);
            break;
        case PIXEL_RGBA2RGB:
        case PIXEL_BGRA2BGR:
            from_rgba2rgb(pixels, w, h, stride, m);
            break;
        case PIXEL_RGBA2BGR:
        case PIXEL_BGRA2RGB:
            from_rgba2bgr(pixels, w, h, stride, m);
            break;
        case PIXEL_RGBA2GRAY:
            from_rgba2gray(pixels, w, h, stride, m);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            from_rgba2bgra(pixels, w, h, stride, m);
            break;
        case PIXEL_BGRA2GRAY:
            from_bgra2gray(pixels, w, h, stride, m);
            break;
        default:
            // unimplemented convert type
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            from_rgb(pixels, w, h, stride, m);

        if (type == PIXEL_GRAY)
            from_gray(pixels, w, h, stride, m);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            from_rgba(pixels, w, h, stride, m);
    }

    return m;
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 3, target_width, target_height);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 1, target_width, target_height);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 4, target_width, target_height);
    }

    // unknown convert type
    return Mat();
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height)
{
    if (w == target_width && h == target_height)
        return Mat::from_pixels(pixels, type, w, h, stride);

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        Mat dst(1, target_width, target_height, (size_t)3u, 3);
        resize_bilinear_c3(pixels, w, h, stride, dst, target_width, target_height, target_width * 3);

        return Mat::from_pixels(dst, type, target_width, target_height);
    }
    else if (type_from == PIXEL_GRAY)
    {
        Mat dst(1, target_width, target_height, (size_t)1u, 1);
        resize_bilinear_c1(pixels, w, h, stride, dst, target_width, target_height, target_width * 1);

        return Mat::from_pixels(dst, type, target_width, target_height);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        Mat dst(1, target_width, target_height, (size_t)4u, 4);
        resize_bilinear_c4(pixels, w, h, stride, dst, target_width, target_height, target_width * 4);

        return Mat::from_pixels(dst, type, target_width, target_height);
    }

    // unknown convert type
    return Mat();
}

void Mat::to_pixels(unsigned char* pixels, int type) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels(pixels, type, w * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels(pixels, type, w * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels(pixels, type, w * 4);
    }
}

void Mat::to_pixels(unsigned char* pixels, int type, int stride) const
{
    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            to_bgr2rgb(*this, pixels, stride);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            to_rgb2rgba(*this, pixels, stride);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            to_bgr2rgba(*this, pixels, stride);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            to_gray2rgba(*this, pixels, stride);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            to_rgba2bgra(*this, pixels, stride);
            break;
        default:
            // unimplemented convert type
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            to_rgb(*this, pixels, stride);

        if (type == PIXEL_GRAY)
            to_gray(*this, pixels, stride);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            to_rgba(*this, pixels, stride);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 4);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const
{
    if (w == target_width && h == target_height)
        return to_pixels(pixels, type);

    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        Mat src(w, h, (size_t)3u, 3);

        to_pixels(src, type);

        resize_bilinear_c3(src, w, h, w * 3, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_GRAY)
    {
        Mat src(w, h, (size_t)1u, 1);

        to_pixels(src, type);

        resize_bilinear_c1(src, w, h, w * 1, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        Mat src(w, h, (size_t)4u, 4);

        to_pixels(src, type);

        resize_bilinear_c4(src, w, h, w * 4, pixels, target_width, target_height, target_stride);
    }
}

}    // namespace ncnn
