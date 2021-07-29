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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef __TEST_ONNX_OP_H__
#define __TEST_ONNX_OP_H__

#include "tengine/c_api.h"

#include <fstream>
#include <vector>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <iostream>
#include <cmath>
#include <climits>
#include <cstdio>

#include "onnx.pb.h"

int get_pb_data(float* float_data, const std::string& filepath)
{
    std::ifstream fs(filepath.c_str(), std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath.c_str());
        return -1;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    onnx::TensorProto tp;
    tp.ParseFromCodedStream(&codedstr);

    /* current, only support the type of data is fp32 */
    if (tp.data_type() == 1)
    {
        if (tp.has_raw_data())
        {
            int size = (int)tp.raw_data().size() / 4;
            const float* data = (float*)tp.raw_data().c_str();
            for (int i = 0; i < size; i++)
                float_data[i] = data[i];
        }
        else
        {
            int size = tp.float_data_size();
            const float* data = tp.float_data().data();
            for (int i = 0; i < size; i++)
                float_data[i] = data[i];
        }
    }
    /* current, only support the type of data is fp64 */
    else if (tp.data_type() == 11)
    {
        if (tp.has_raw_data())
        {
            int size = (int)tp.raw_data().size() / 8;
            const double* data = (double*)tp.raw_data().c_str();
            for (int i = 0; i < size; i++)
                float_data[i] = data[i];
        }
        else
        {
            int size = tp.double_data_size();
            const double* data = tp.double_data().data();
            for (int i = 0; i < size; i++)
                float_data[i] = data[i];
        }
    }
    else
    {
        fprintf(stderr, "not support the type of data is %d\n", tp.data_type());
        return -1;
    }

    return 0;
}

int get_pb_data_i32(int32_t* i32_data, const std::string& filepath)
{
    std::ifstream fs(filepath.c_str(), std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath.c_str());
        return -1;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    onnx::TensorProto tp;
    tp.ParseFromCodedStream(&codedstr);

    /* current, only support the type of data is int32 */
    if (tp.data_type() == 6)
    {
        if (tp.has_raw_data())
        {
            int size = (int)tp.raw_data().size() / 4;
            const int32_t* data = (int32_t*)tp.raw_data().c_str();
            for (int i = 0; i < size; i++)
                i32_data[i] = data[i];
        }
        else
        {
            int size = tp.int32_data_size();
            const int32_t* data = tp.int32_data().data();
            for (int i = 0; i < size; i++)
                i32_data[i] = data[i];
        }
    }
    else if (tp.data_type() == 7)
    {
        if (tp.has_raw_data())
        {
            int size = (int)tp.raw_data().size() / 8;
            const int64_t* data = (int64_t*)tp.raw_data().c_str();
            for (int i = 0; i < size; i++)
                i32_data[i] = (int32_t)data[i];
        }
        else
        {
            int size = tp.int64_data_size();
            const int64_t* data = tp.int64_data().data();
            for (int i = 0; i < size; i++)
                i32_data[i] = (int32_t)data[i];
        }
    }
    else
    {
        fprintf(stderr, "not support the type of data is %d\n", tp.data_type());
        return -1;
    }

    return 0;
}

int float_mismatch(float* current, float* reference, int size)
{
    int ret = 0;
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        if (fabs(tmp) > 0.0001)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            ret = -1;
        }
    }
    fprintf(stderr, "test pass\n");

    return ret;
}

#endif
