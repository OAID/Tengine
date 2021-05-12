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
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#ifndef VULKAN_TENSOR_HPP
#define VULKAN_TENSOR_HPP

#include <iostream>
#include <cstring>
// #include "tengine_ir.h"

extern "C"
{
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

#include <vulkan/vulkan.h>
#include "vulkan_allocator.hpp"
#include "vulkan_option.hpp"

namespace TEngine {

class VkTensor;
class VkImageTensor;

class Tshape
{
public:
    Tshape()
    {
        w = 0;
        h = 0;
        c = 0;
        dims = 0;
    }
    Tshape(int _w, int _h, int _c)
    {
        w = _w;
        h = _h;
        c = _c;
        dims = 3;
    }

    int dims;
    int w;
    int h;
    int c;

    size_t cstep;
};

class Tensor
{
public:
    // empty
    Tensor();
    // vec
    Tensor(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // image
    Tensor(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // dim
    Tensor(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // packed vec
    Tensor(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed image
    Tensor(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed dim
    Tensor(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // copy
    Tensor(const Tensor& m);
    // copy from ir_tensor
    Tensor(struct tensor* m);
    // external vec
    Tensor(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external image
    Tensor(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external dim
    Tensor(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external packed vec
    Tensor(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed image
    Tensor(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed dim
    Tensor(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // release
    ~Tensor();
    // assign
    Tensor& operator=(const Tensor& m);

    // reshape vec
    Tensor reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    Tensor reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    Tensor reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate like
    void create_like(const tensor* m, Allocator* allocator = 0);
    // allocate like
    void create_like(const Tensor& m, Allocator* allocator = 0);
    // allocate like
    void create_like(const VkTensor& m, Allocator* allocator = 0);
    // allocate like
    void create_like(const VkImageTensor& im, Allocator* allocator = 0);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // shape only
    Tensor shape() const;

    // data reference
    Tensor channel(int c);
    const Tensor channel(int c) const;
    float* row(int y);
    const float* row(int y) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};



class VkTensor
{
public:
    // empty
    VkTensor();
    // vec
    VkTensor(int w, size_t elemsize, VkAllocator* allocator);
    // image
    VkTensor(int w, int h, size_t elemsize, VkAllocator* allocator);
    // dim
    VkTensor(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // packed vec
    VkTensor(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed image
    VkTensor(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed dim
    VkTensor(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // copy
    VkTensor(const VkTensor& m);
    // external vec
    VkTensor(int w, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external image
    VkTensor(int w, int h, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external dim
    VkTensor(int w, int h, int c, VkBufferMemory* data, size_t elemsize, VkAllocator* allocator);
    // external packed vec
    VkTensor(int w, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed image
    VkTensor(int w, int h, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed dim
    VkTensor(int w, int h, int c, VkBufferMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // release
    ~VkTensor();
    // assign
    VkTensor& operator=(const VkTensor& m);
        // reshape vec
    VkTensor reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    VkTensor reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    VkTensor reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate like
    void create_like(const Tensor& m, VkAllocator* allocator);
    void create_like(const tensor* m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkTensor& m, VkAllocator* allocator);

    // allocate vec
    void create(struct tensor* tensor, VkAllocator* allocator);

    // staging buffer
    void prepare_staging_buffer();
    void discard_staging_buffer();

    // copy
    // void upload(const Tensor& m);
    // void download(Tensor& m) const;

    // mapped
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // shape only
    // Mat shape() const;
    
    // low-level reference
    VkBuffer buffer() const;
    size_t buffer_offset() const;
    size_t buffer_capacity() const;

    // device buffer
    VkBufferMemory* data;

    // staging buffer
    VkBufferMemory* staging_data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;
    int* staging_refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;
    VkAllocator* staging_allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

class VkImageTensor
{
public:
    // empty
    VkImageTensor();
    // vec
    VkImageTensor(int w, size_t elemsize, VkAllocator* allocator);
    // image
    VkImageTensor(int w, int h, size_t elemsize, VkAllocator* allocator);
    // dim
    VkImageTensor(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // packed vec
    VkImageTensor(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed image
    VkImageTensor(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // packed dim
    VkImageTensor(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // copy
    VkImageTensor(const VkImageTensor& m);
    // external vec
    VkImageTensor(int w, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external image
    VkImageTensor(int w, int h, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external dim
    VkImageTensor(int w, int h, int c, VkImageMemory* data, size_t elemsize, VkAllocator* allocator);
    // external packed vec
    VkImageTensor(int w, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed image
    VkImageTensor(int w, int h, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // external packed dim
    VkImageTensor(int w, int h, int c, VkImageMemory* data, size_t elemsize, int elempack, VkAllocator* allocator);
    // release
    ~VkImageTensor();
    // assign
    VkImageTensor& operator=(const VkImageTensor& m);
    // allocate vec
    void create(int w, size_t elemsize, VkAllocator* allocator);
    // allocate image
    void create(int w, int h, size_t elemsize, VkAllocator* allocator);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize, VkAllocator* allocator);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, VkAllocator* allocator);
    // allocate like
    void create_like(const tensor* m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkTensor& m, VkAllocator* allocator);
    // allocate like
    void create_like(const VkImageTensor& im, VkAllocator* allocator);


    // mapped
    ///Mat mapped() const;
    void* mapped_ptr() const;

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // shape only
    ///Mat shape() const;

    // low-level reference
    VkImage image() const;
    VkImageView imageview() const;

#if __ANDROID_API__ >= 26
    // convenient construct from android hardware buffer
    static VkImageMat from_android_hardware_buffer(VkAndroidHardwareBufferImageAllocator* allocator);
#endif // __ANDROID_API__ >= 26

    // device image
    VkImageMemory* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    VkAllocator* allocator;

    // the dimension rank
    int dims;

    int w;
    int h;
    int c;
};

inline VkTensor::VkTensor()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline VkTensor::VkTensor(int _w, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

inline VkTensor::VkTensor(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline VkTensor::VkTensor(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline VkTensor::VkTensor(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline VkTensor::VkTensor(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline VkTensor::VkTensor(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline VkTensor::VkTensor(const VkTensor& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c)
{
    if (refcount)
        TENGINE_XADD(refcount, 1);

    cstep = m.cstep;
}

inline VkTensor::VkTensor(int _w, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkTensor::VkTensor(int _w, int _h, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkTensor::VkTensor(int _w, int _h, int _c, VkBufferMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkTensor::VkTensor(int _w, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline VkTensor::VkTensor(int _w, int _h, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline VkTensor::VkTensor(int _w, int _h, int _c, VkBufferMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline VkTensor::~VkTensor()
{
    release();
}

inline VkTensor& VkTensor::operator=(const VkTensor& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        TENGINE_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void VkTensor::create(int _w, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    // cstep = alignSize(w * h * elemsize, 16) / elemsize;
    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

inline void VkTensor::create_like(const tensor* m, VkAllocator* _allocator)
{
    int _c = m->dims[1];
    int _h = m->dims[2];
    int _w = m->dims[3];
    size_t _elemsize = m->data_type == 0 ? 4 : 1;
    int _elempack = 1;

    if (_c == 0 && _h == 0 && _w != 0)
        create(_w, _elemsize, _elempack, _allocator);
    if (_c == 0 && _h != 0 && _w != 0)
        create(_w, _h, _elemsize, _elempack, _allocator);
    if (_c != 0 && _h != 0 && _w != 0)
        create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline void VkTensor::create_like(const Tensor& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void VkTensor::create_like(const VkTensor& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void* VkTensor::mapped_ptr() const
{
    if (!allocator->mappable)
        return 0;

    return (unsigned char*)data->mapped_ptr + data->offset;
}

inline void VkTensor::addref()
{
    if (refcount)
        TENGINE_XADD(refcount, 1);
}

inline void VkTensor::release()
{
    if (refcount && TENGINE_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool VkTensor::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t VkTensor::total() const
{
    return cstep * c;
}

// TODO
// inline Mat VkTensor::shape() const
// {
//     if (dims == 1)
//         return Mat(w * elempack, (void*)0);
//     if (dims == 2)
//         return Mat(w, h * elempack, (void*)0);
//     if (dims == 3)
//         return Mat(w, h, c * elempack, (void*)0);

//     return Mat();
// }

inline VkBuffer VkTensor::buffer() const
{
    return data->buffer;
}

inline size_t VkTensor::buffer_offset() const
{
    return data->offset;
}

inline size_t VkTensor::buffer_capacity() const
{
    return data->capacity;
}

// VkImageTensor
inline VkImageTensor::VkImageTensor()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
}

inline VkImageTensor::VkImageTensor(int _w, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _elemsize, _allocator);
}

inline VkImageTensor::VkImageTensor(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline VkImageTensor::VkImageTensor(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline VkImageTensor::VkImageTensor(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline VkImageTensor::VkImageTensor(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline VkImageTensor::VkImageTensor(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline VkImageTensor::VkImageTensor(const VkImageTensor& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c)
{
    if (refcount)
        TENGINE_XADD(refcount, 1);
}

inline VkImageTensor::VkImageTensor(int _w, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
}

inline VkImageTensor::VkImageTensor(int _w, int _h, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
}

inline VkImageTensor::VkImageTensor(int _w, int _h, int _c, VkImageMemory* _data, size_t _elemsize, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
}

inline VkImageTensor::VkImageTensor(int _w, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
}

inline VkImageTensor::VkImageTensor(int _w, int _h, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
}

inline VkImageTensor::VkImageTensor(int _w, int _h, int _c, VkImageMemory* _data, size_t _elemsize, int _elempack, VkAllocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
}

inline VkImageTensor::~VkImageTensor()
{
    release();
}

inline VkImageTensor& VkImageTensor::operator=(const VkImageTensor& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        TENGINE_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    return *this;
}

inline void VkImageTensor::create(int _w, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    if (total() > 0)
    {
        data = allocator->fastMalloc(dims, w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

inline void VkImageTensor::create_like(const tensor* m, VkAllocator* _allocator)
{
    int _c = m->dims[1];
    int _h = m->dims[2];
    int _w = m->dims[3];
    size_t _elemsize = m->data_type == 0 ? 4 : 1;
    int _elempack = 1;
    int _dims = m->dim_num;

    if (_dims == 1)
        create(_w, _elemsize, _elempack, _allocator);
    if (_dims == 2)
        create(_w, _h, _elemsize, _elempack, _allocator);
    if (_dims == 3)
        create(_w, _h, _c, _elemsize, _elempack, _allocator);
}


inline void VkImageTensor::create_like(const VkTensor& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void VkImageTensor::create_like(const VkImageTensor& im, VkAllocator* _allocator)
{
    int _dims = im.dims;
    if (_dims == 1)
        create(im.w, im.elemsize, im.elempack, _allocator);
    if (_dims == 2)
        create(im.w, im.h, im.elemsize, im.elempack, _allocator);
    if (_dims == 3)
        create(im.w, im.h, im.c, im.elemsize, im.elempack, _allocator);
}

// inline Mat VkImageMat::mapped() const
// {
//     if (!allocator->mappable || !data->mapped_ptr)
//         return Mat();

//     if (dims == 1)
//         return Mat(w, mapped_ptr(), elemsize, elempack, 0);

//     if (dims == 2)
//         return Mat(w, h, mapped_ptr(), elemsize, elempack, 0);

//     if (dims == 3)
//         return Mat(w, h, c, mapped_ptr(), elemsize, elempack, 0);

//     return Mat();
// }

inline void* VkImageTensor::mapped_ptr() const
{
    if (!allocator->mappable || !data->mapped_ptr)
        return 0;

    return (unsigned char*)data->mapped_ptr + data->bind_offset;
}

inline void VkImageTensor::addref()
{
    if (refcount)
        TENGINE_XADD(refcount, 1);
}

inline void VkImageTensor::release()
{
    if (refcount && TENGINE_XADD(refcount, -1) == 1)
    {
        if (allocator && data)
        {
            allocator->fastFree(data);
        }
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    refcount = 0;
}

inline bool VkImageTensor::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t VkImageTensor::total() const
{
    return w * h * c;
}

// inline Mat VkImageTensor::shape() const
// {
//     if (dims == 1)
//         return Mat(w * elempack, (void*)0);
//     if (dims == 2)
//         return Mat(w, h * elempack, (void*)0);
//     if (dims == 3)
//         return Mat(w, h, c * elempack, (void*)0);

//     return Mat();
// }

inline VkImage VkImageTensor::image() const
{
    return data->image;
}

inline VkImageView VkImageTensor::imageview() const
{
    return data->imageview;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Tensor defination

inline Tensor::Tensor()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}   

inline Tensor::Tensor(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}   

inline Tensor::Tensor(int _w, int _h, size_t _elemsize, Allocator* _allocator)     : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0){
    create(_w, _h, _elemsize, _allocator);}
inline Tensor::Tensor(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline Tensor::Tensor(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline Tensor::Tensor(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline Tensor::Tensor(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline Tensor::Tensor(const Tensor& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c), cstep(m.cstep)
{
    if (refcount)
        TENGINE_XADD(refcount, 1);
}

inline Tensor::Tensor(struct tensor* m)
    : data(m->data), refcount(0), elemsize(0), elempack(1), allocator(0), dims(0), w(0), h(0), c(0)
{
    if(m->layout == 0)
    {
        c = m->dims[1];
        h = m->dims[2];
        w = m->dims[3];
        elemsize = m->elem_size;
        elempack = 1;
        dims = 3;
        cstep = w * h;
    }
    else
    {
        c = m->dims[3];
        h = m->dims[2];
        w = m->dims[1];
        elemsize = m->elem_size;
        elempack = 1;
        dims = 3;
        cstep = w * h;
    }
}
inline Tensor::Tensor(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Tensor::Tensor(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Tensor::Tensor(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Tensor::Tensor(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Tensor::Tensor(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Tensor::Tensor(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Tensor::~Tensor()
{
    release();
}

inline Tensor& Tensor::operator=(const Tensor& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        TENGINE_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline Tensor Tensor::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * c != _w)
        return Tensor();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Tensor m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Tensor m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

inline Tensor Tensor::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * c != _w * _h)
        return Tensor();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Tensor m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Tensor m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = _w * _h;

    return m;
}

inline Tensor Tensor::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * c != _w * _h * _c)
        return Tensor();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize(_w * _h * elemsize, 16) / elemsize)
        {
            Tensor m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i=0; i<_c; i++)
            {
                const void* ptr = (unsigned char*)data + i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, _w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Tensor tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Tensor m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize(_w * _h * elemsize, 16) / elemsize;

    return m;
}

inline void Tensor::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Tensor::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Tensor::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Tensor::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Tensor::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Tensor::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = w * h;    //alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

// inline void Tensor::create_like(const tensor* m, Allocator* _allocator)
// {
//     int _dims = m.dims;
//     if (_dims == 1)
//         create(m.w, m.elemsize, m.elempack, _allocator);
//     if (_dims == 2)
//         create(m.w, m.h, m.elemsize, m.elempack, _allocator);
//     if (_dims == 3)
//         create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
// }

inline void Tensor::create_like(const Tensor& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void Tensor::create_like(const VkTensor& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void Tensor::create_like(const VkImageTensor& im, Allocator* _allocator)
{
    int _dims = im.dims;
    if (_dims == 1)
        create(im.w, im.elemsize, im.elempack, _allocator);
    if (_dims == 2)
        create(im.w, im.h, im.elemsize, im.elempack, _allocator);
    if (_dims == 3)
        create(im.w, im.h, im.c, im.elemsize, im.elempack, _allocator);
}

inline void Tensor::addref()
{
    if (refcount)
        TENGINE_XADD(refcount, 1);
}

inline void Tensor::release()
{
    if (refcount && TENGINE_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool Tensor::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Tensor::total() const
{
    return cstep * c;
}

inline Tensor Tensor::shape() const
{
    if (dims == 1)
        return Tensor(w * elempack, (void*)0);
    if (dims == 2)
        return Tensor(w, h * elempack, (void*)0);
    if (dims == 3)
        return Tensor(w, h, c * elempack, (void*)0);

    return Tensor();
}

inline Tensor Tensor::channel(int _c)
{
    return Tensor(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Tensor Tensor::channel(int _c) const
{
    return Tensor(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline float* Tensor::row(int y)
{
    return (float*)((unsigned char*)data + w * y * elemsize);
}

inline const float* Tensor::row(int y) const
{
    return (const float*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline Tensor::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Tensor::operator const T*() const
{
    return (const T*)data;
}

void convert_packing(const Tensor& src, Tensor& dst, int elempack, const Option& opt = Option());
void convert_packing(tensor* src, Tensor&dst, int elempack, const Option& opt = Option());
void cast_float32_to_float16(const Tensor& src, Tensor& dst, const Option& opt = Option());
void cast_float16_to_float32(const Tensor& src, Tensor& dst, const Option& opt = Option());


} // namespace TEngine


#endif // VULKAN_TENSOR_HPP

