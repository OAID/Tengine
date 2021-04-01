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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#pragma once

#include <memory>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>


#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                                       \
    do                                                                      \
    {                                                                       \
        auto ret = (status);                                                \
        if (ret != 0)                                                       \
        {                                                                   \
            Log(Loglevel, "TensorRT Engine",  "Cuda failure: %d", ret);     \
            abort();                                                        \
        }                                                                   \
    } while (0)


constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KiB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val) { return val * (1 << 30); }
constexpr long long int operator"" _MiB(long long unsigned int val) { return val * (1 << 20); }
constexpr long long int operator"" _KiB(long long unsigned int val) { return val * (1 << 10); }


char* ReplaceSubStr(char* str, char* srcSubStr, char* dstSubStr, char* out)
{
    char* p;
    char* _out = out;
    char* _str = str;
    char* _src = srcSubStr;
    char* _dst = dstSubStr;
    int src_size = strlen(_src);
    int dst_size = strlen(_dst);
    int len = 0;

    do
    {
        p = strstr(_str, _src);
        if (p == 0)
        {
            strcpy(_out, _str);
            return out;
        }
        len = p - _str;
        memcpy(_out, _str, len);
        memcpy(_out + len, _dst, dst_size);
        _str = p + src_size;
        _out = _out + len + dst_size;
    } while (p);

    return out;
}


void extract_feature_blob_f32(char* comment, char* layer_name, const struct ir_tensor* tensor)
{
    char file_path_output[2048] = {'\0'};
    char file_dir[128] = {'\0'};
    int type = tensor->data_type;

    FILE* pFile = NULL;

    char name[1024];

    ReplaceSubStr(layer_name, "/", "-", name);

    sprintf(file_dir, "./output/");

#ifdef _MSC_VER
    CreateDirectoryA(file_dir, NULL);
#else
    mkdir(file_dir, 0777);
#endif
    sprintf(file_path_output, "./output/%s_%s_blob_data.txt", name, comment);

    pFile = fopen(file_path_output, "w");
    if (pFile == NULL)
    {
        fprintf(stderr, "open file error!\n");
    }

    if (type == TENGINE_DT_FP32)
    {
        switch (tensor->dim_num)
        {
            case 5: {
                int dim5 = tensor->dims[0], batch = tensor->dims[1], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[2];
                    height = tensor->dims[3];
                    width = tensor->dims[4];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                    channel = tensor->dims[4];
                }

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is fp32\n", dim5, batch, channel, height, width);

                float* base_ptr = (float*)tensor->data;

                for (int d5 = 0; d5 < dim5; d5++)
                {
                    fprintf(pFile, "Dim5 %d:\n", d5);

                    for (int n = 0; n < batch; n++)
                    {
                        fprintf(pFile, "\tBatch %d:\n", n);

                        for (int ch = 0; ch < channel; ch++)
                        {
                            fprintf(pFile, "\t\tChannel %d:\n", ch);

                            for (int h = 0; h < height; h++)
                            {
                                fprintf(pFile, "\t\t\t");

                                for (int w = 0; w < width; w++)
                                {
                                    int offset = 0;
                                    if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch * height * width;
                                        offset += h * width;
                                        offset += w;
                                    }
                                    if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch;
                                        offset += h * width * channel;
                                        offset += w * channel;
                                    }

                                    float val = base_ptr[offset];
                                    if (val < 0)
                                        fprintf(pFile, "%.4f ", val);
                                    else
                                        fprintf(pFile, " %.4f ", val);
                                }
                                fprintf(pFile, "\n");
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 4: {
                int batch = tensor->dims[0], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[1];
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[1];
                    width = tensor->dims[2];
                    channel = tensor->dims[3];
                }

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is fp32\n", batch, channel, height, width);

                float* base_ptr = (float*)tensor->data;
                for (int n = 0; n < batch; n++)
                {
                    fprintf(pFile, "Batch %d:\n", n);

                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(pFile, "\tChannel %d:\n", ch);

                        for (int h = 0; h < height; h++)
                        {
                            fprintf(pFile, "\t\t");

                            for (int w = 0; w < width; w++)
                            {
                                int offset = 0;
                                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch * height * width;
                                    offset += h * width;
                                    offset += w;
                                }
                                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch;
                                    offset += h * width * channel;
                                    offset += w * channel;
                                }

                                float val = base_ptr[offset];
                                if (val < 0)
                                    fprintf(pFile, "%.4f ", val);
                                else
                                    fprintf(pFile, " %.4f ", val);
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 3: {
                int channel = 0, height = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[0];
                    height = tensor->dims[1];
                    width = tensor->dims[2];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                    channel = tensor->dims[2];
                }

                fprintf(pFile, "Shape is {%d %d %d}, data type is fp32\n", channel, height, width);

                float* base_ptr = (float*)tensor->data;
                for (int ch = 0; ch < channel; ch++)
                {
                    fprintf(pFile, "Channel %d:\n", ch);

                    for (int h = 0; h < height; h++)
                    {
                        fprintf(pFile, "\t");

                        for (int w = 0; w < width; w++)
                        {
                            int offset = 0;

                            if (TENGINE_LAYOUT_NCHW == tensor->layout)
                            {
                                offset += ch * height * width;
                                offset += h * width;
                                offset += w;
                            }
                            if (TENGINE_LAYOUT_NHWC == tensor->layout)
                            {
                                offset += ch;
                                offset += h * width * channel;
                                offset += w * channel;
                            }

                            float val = base_ptr[offset];
                            if (val < 0)
                                fprintf(pFile, "%.4f ", val);
                            else
                                fprintf(pFile, " %.4f ", val);
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 2: {
                int height = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }

                fprintf(pFile, "Shape is {%d %d}, data type is fp32\n", height, width);

                float* base_ptr = (float*)tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        float val = base_ptr[offset];
                        if (val < 0)
                            fprintf(pFile, "%.4f ", val);
                        else
                            fprintf(pFile, " %.4f ", val);
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 1: {
                int width = tensor->dims[0];

                fprintf(pFile, "Shape is {%d}, data type is fp32\n", width);

                float* base_ptr = (float*)tensor->data;

                for (int w = 0; w < width; w++)
                {
                    float val = base_ptr[w];
                    if (val < 0)
                        fprintf(pFile, "%.4f ", val);
                    else
                        fprintf(pFile, " %.4f ", val);
                }

                break;
            }
        }
    }
    else if (type == TENGINE_DT_UINT8)
    {
        float scale = tensor->scale;
        int32_t zero_point = tensor->zero_point;
        /* dequant to fp32 */
        switch (tensor->dim_num)
        {
            case 5: {
                int dim5 = tensor->dims[0], batch = tensor->dims[1], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[2];
                    height = tensor->dims[3];
                    width = tensor->dims[4];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                    channel = tensor->dims[4];
                }

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is uint8, dequant to fp32\n", dim5, batch, channel, height, width);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int d5 = 0; d5 < dim5; d5++)
                {
                    fprintf(pFile, "Dim5 %d:\n", d5);

                    for (int n = 0; n < batch; n++)
                    {
                        fprintf(pFile, "\tBatch %d:\n", n);

                        for (int ch = 0; ch < channel; ch++)
                        {
                            fprintf(pFile, "\t\tChannel %d:\n", ch);

                            for (int h = 0; h < height; h++)
                            {
                                fprintf(pFile, "\t\t\t");

                                for (int w = 0; w < width; w++)
                                {
                                    int offset = 0;
                                    if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch * height * width;
                                        offset += h * width;
                                        offset += w;
                                    }
                                    if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch;
                                        offset += h * width * channel;
                                        offset += w * channel;
                                    }

                                    unsigned char val = base_ptr[offset];
                                    float val_fp32 = (val - zero_point) * scale;
                                    if (val_fp32 < 0)
                                        fprintf(pFile, "%.4f ", val_fp32);
                                    else
                                        fprintf(pFile, " %.4f ", val_fp32);
                                }
                                fprintf(pFile, "\n");
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 4: {
                int batch = tensor->dims[0], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[1];
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[1];
                    width = tensor->dims[2];
                    channel = tensor->dims[3];
                }

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is uint8\n", batch, channel, height, width);

                unsigned char* base_ptr = (unsigned char*)tensor->data;
                for (int n = 0; n < batch; n++)
                {
                    fprintf(pFile, "Batch %d:\n", n);

                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(pFile, "\tChannel %d:\n", ch);

                        for (int h = 0; h < height; h++)
                        {
                            fprintf(pFile, "\t\t");

                            for (int w = 0; w < width; w++)
                            {
                                int offset = 0;
                                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch * height * width;
                                    offset += h * width;
                                    offset += w;
                                }
                                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch;
                                    offset += h * width * channel;
                                    offset += w * channel;
                                }

                                unsigned char val = base_ptr[offset];
                                float val_fp32 = (val - zero_point) * scale;
                                if (val_fp32 < 0)
                                    fprintf(pFile, "%.4f ", val_fp32);
                                else
                                    fprintf(pFile, " %.4f ", val_fp32);
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 3: {
                int batch = 0, channel = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    batch = tensor->dims[0];
                    channel = tensor->dims[1];
                    width = tensor->dims[2];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    batch = tensor->dims[0];
                    width = tensor->dims[1];
                    channel = tensor->dims[2];
                }

                fprintf(pFile, "Shape is {%d %d %d}, data type is uint8\n", batch, channel, width);

                unsigned char* base_ptr = (unsigned char*)tensor->data;
                for (int n = 0; n < batch; n++)
                {
                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(pFile, "Channel %d:\n", ch);
                        fprintf(pFile, "\t");

                        for (int w = 0; w < width; w++)
                        {
                            int offset = 0;

                            if (TENGINE_LAYOUT_NCHW == tensor->layout)
                            {
                                offset += n * channel * width;
                                offset += ch * width;
                                offset += w;
                            }
                            if (TENGINE_LAYOUT_NHWC == tensor->layout)
                            {
                                offset += ch;
                                offset += n * width * channel;
                                offset += w * channel;
                            }

                            unsigned char val = base_ptr[offset];
                            float val_fp32 = (val - zero_point) * scale;
                            if (val_fp32 < 0)
                                fprintf(pFile, "%.4f ", val_fp32);
                            else
                                fprintf(pFile, " %.4f ", val_fp32);
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 2: {
                int height = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }

                fprintf(pFile, "Shape is {%d %d}, data type is uint8\n", height, width);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        unsigned char val = base_ptr[offset];
                        float val_fp32 = (val - zero_point) * scale;
                        if (val_fp32 < 0)
                            fprintf(pFile, "%.4f ", val_fp32);
                        else
                            fprintf(pFile, " %.4f ", val_fp32);
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 1: {
                int width = tensor->dims[0];

                fprintf(pFile, "Shape is {%d}, data type is uint8\n", width);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int w = 0; w < width; w++)
                {
                    unsigned char val = base_ptr[w];
                    float val_fp32 = (val - zero_point) * scale;
                    if (val_fp32 < 0)
                        fprintf(pFile, "%.4f ", val_fp32);
                    else
                        fprintf(pFile, " %.4f ", val_fp32);
                }

                break;
            }
        }

        /* original uint8 */
        fprintf(pFile, "\n\n");
        switch (tensor->dim_num)
        {
            case 5: {
                int dim5 = tensor->dims[0], batch = tensor->dims[1], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[2];
                    height = tensor->dims[3];
                    width = tensor->dims[4];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                    channel = tensor->dims[4];
                }

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is uint8, scale %f, zero_point %d\n", dim5, batch, channel, height, width, scale, zero_point);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int d5 = 0; d5 < dim5; d5++)
                {
                    fprintf(pFile, "Dim5 %d:\n", d5);

                    for (int n = 0; n < batch; n++)
                    {
                        fprintf(pFile, "\tBatch %d:\n", n);

                        for (int ch = 0; ch < channel; ch++)
                        {
                            fprintf(pFile, "\t\tChannel %d:\n", ch);

                            for (int h = 0; h < height; h++)
                            {
                                fprintf(pFile, "\t\t\t");

                                for (int w = 0; w < width; w++)
                                {
                                    int offset = 0;
                                    if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch * height * width;
                                        offset += h * width;
                                        offset += w;
                                    }
                                    if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                    {
                                        offset += d5 * batch * channel * height * width;
                                        offset += n * channel * height * width;
                                        offset += ch;
                                        offset += h * width * channel;
                                        offset += w * channel;
                                    }

                                    unsigned char val = base_ptr[offset];

                                    fprintf(pFile, "%3d ", val);
                                }
                                fprintf(pFile, "\n");
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 4: {
                int batch = tensor->dims[0], channel = 0, height = 0, width = 0;
                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    channel = tensor->dims[1];
                    height = tensor->dims[2];
                    width = tensor->dims[3];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[1];
                    width = tensor->dims[2];
                    channel = tensor->dims[3];
                }

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is uint8, scale %f, zero_point %d\n", batch, channel, height, width, scale, zero_point);

                unsigned char* base_ptr = (unsigned char*)tensor->data;
                for (int n = 0; n < batch; n++)
                {
                    fprintf(pFile, "Batch %d:\n", n);

                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(pFile, "\tChannel %d:\n", ch);

                        for (int h = 0; h < height; h++)
                        {
                            fprintf(pFile, "\t\t");

                            for (int w = 0; w < width; w++)
                            {
                                int offset = 0;
                                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch * height * width;
                                    offset += h * width;
                                    offset += w;
                                }
                                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                                {
                                    offset += n * channel * height * width;
                                    offset += ch;
                                    offset += h * width * channel;
                                    offset += w * channel;
                                }

                                unsigned char val = base_ptr[offset];

                                fprintf(pFile, "%3d ", val);
                            }
                            fprintf(pFile, "\n");
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 3: {
                int batch = 0, channel = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    batch = tensor->dims[0];
                    channel = tensor->dims[1];
                    width = tensor->dims[2];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    batch = tensor->dims[0];
                    width = tensor->dims[1];
                    channel = tensor->dims[2];
                }

                fprintf(pFile, "Shape is {%d %d %d}, data type is uint8, scale %f, zero_point %d\n", batch, channel, width, scale, zero_point);

                unsigned char* base_ptr = (unsigned char*)tensor->data;
                for (int n = 0; n < batch; n++)
                {
                    for (int ch = 0; ch < channel; ch++)
                    {
                        fprintf(pFile, "Channel %d:\n", ch);
                        fprintf(pFile, "\t");

                        for (int w = 0; w < width; w++)
                        {
                            int offset = 0;

                            if (TENGINE_LAYOUT_NCHW == tensor->layout)
                            {
                                offset += n * channel * width;
                                offset += ch * width;
                                offset += w;
                            }
                            if (TENGINE_LAYOUT_NHWC == tensor->layout)
                            {
                                offset += ch;
                                offset += n * width * channel;
                                offset += w * channel;
                            }

                            unsigned char val = base_ptr[offset];

                            fprintf(pFile, "%3d ", val);
                        }
                        fprintf(pFile, "\n");
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 2: {
                int height = 0, width = 0;

                if (TENGINE_LAYOUT_NCHW == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }
                if (TENGINE_LAYOUT_NHWC == tensor->layout)
                {
                    height = tensor->dims[0];
                    width = tensor->dims[1];
                }

                fprintf(pFile, "Shape is {%d %d}, data type is uint8, scale %f, zero_point %d\n", height, width, scale, zero_point);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        unsigned char val = base_ptr[offset];

                        fprintf(pFile, "%3d ", val);
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 1: {
                int width = tensor->dims[0];

                fprintf(pFile, "Shape is {%d}, data type is uint8, scale %f, zero_point %d\n", width, scale, zero_point);

                unsigned char* base_ptr = (unsigned char*)tensor->data;

                for (int w = 0; w < width; w++)
                {
                    unsigned char val = base_ptr[w];

                    fprintf(pFile, "%3d ", val);
                }

                break;
            }
        }
    }
    else
    {
        printf("Input data type %d not to be supported.\n", type);
    }

    // close file
    fclose(pFile);
    pFile = NULL;
}




