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
 * Author: haitao@openailab.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "sys_port.h"
#include "tengine_errno.h"
#include "tengine_utils.h"
#include "tengine_ir.h"
#include "nn_device.h"
#include "cpu_device.h"
#include "cpu_node_ops.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "compiler_fp16.h"

#include <sys/time.h>

#include <stdlib.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#ifdef DEBUG_DATA
char* ReplaceSubStr(const char* str, const char* srcSubStr, const char* dstSubStr, char* out)
{
    char* p;
    char* _out = out;
    const char* _str = str;
    const char* _src = srcSubStr;
    const char* _dst = dstSubStr;
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

/*
 * Extract the blob feature map
 */
void extract_feature_blob_f32(const char* comment, const char* layer_name, const struct ir_tensor* tensor)
{
    char file_path_output[512] = {'\0'};
    char file_dir[128] = {'\0'};
    int type = tensor->data_type;

    FILE* pFile = NULL;

    char name[1024];

    ReplaceSubStr(layer_name, "/", "-", name);

    sprintf(file_dir, "./output/");
    mkdir(file_dir, 0777);

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

                float* base_ptr = tensor->data;

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

                float* base_ptr = tensor->data;
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

                float* base_ptr = tensor->data;
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

                float* base_ptr = tensor->data;

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

                float* base_ptr = tensor->data;

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
    else if (type == TENGINE_DT_FP16)
    {
        /* cast fp16 to fp32 */
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

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is fp16, cost to fp32\n", dim5, batch, channel, height, width);

                __fp16* base_ptr = tensor->data;

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

                                    __fp16 val = base_ptr[offset];
                                    float val_fp32 = fp16_to_fp32(val);
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

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is fp16, cost to fp32\n", batch, channel, height, width);

                __fp16* base_ptr = tensor->data;
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

                                __fp16 val = base_ptr[offset];
                                float val_fp32 = fp16_to_fp32(val);
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

                fprintf(pFile, "Shape is {%d %d %d}, data type is fp16, cost to fp32\n", batch, channel, width);

                __fp16* base_ptr = tensor->data;
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

                            __fp16 val = base_ptr[offset];
                            float val_fp32 = fp16_to_fp32(val);
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

                fprintf(pFile, "Shape is {%d %d}, data type is fp16, cost to fp32\n", height, width);

                __fp16* base_ptr = tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        __fp16 val = base_ptr[offset];
                        float val_fp32 = fp16_to_fp32(val);
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

                fprintf(pFile, "Shape is {%d}, data type is fp16, cost to fp32\n", width);

                __fp16* base_ptr = tensor->data;

                for (int w = 0; w < width; w++)
                {
                    __fp16 val = base_ptr[w];
                    float val_fp32 = fp16_to_fp32(val);
                    if (val_fp32 < 0)
                        fprintf(pFile, "%.4f ", val_fp32);
                    else
                        fprintf(pFile, " %.4f ", val_fp32);
                }

                break;
            }
        }
    }    
    else if(type == TENGINE_DT_UINT8)
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

                unsigned char* base_ptr = tensor->data;

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

                unsigned char* base_ptr = tensor->data;
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

                unsigned char* base_ptr = tensor->data;
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

                unsigned char* base_ptr = tensor->data;

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

                unsigned char* base_ptr = tensor->data;

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

                unsigned char* base_ptr = tensor->data;

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

                unsigned char* base_ptr = tensor->data;
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

                unsigned char* base_ptr = tensor->data;
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

                unsigned char* base_ptr = tensor->data;

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

                unsigned char* base_ptr = tensor->data;

                for (int w = 0; w < width; w++)
                {
                    unsigned char val = base_ptr[w];

                    fprintf(pFile, "%3d ", val);
                }

                break;
            }
        }
    }
    else if (type == TENGINE_DT_INT8)
    {
        float scale = tensor->scale;
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

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is int8, dequant to fp32\n", dim5, batch, channel, height, width);

                int8_t * base_ptr = tensor->data;

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

                                    int8_t val = base_ptr[offset];
                                    float val_fp32 = (float )val * scale;
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

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is int8\n", batch, channel, height, width);

                int8_t * base_ptr = tensor->data;
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

                                int8_t val = base_ptr[offset];
                                float val_fp32 = (float )val * scale;
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

                fprintf(pFile, "Shape is {%d %d %d}, data type is int8\n", batch, channel, width);

                int8_t * base_ptr = tensor->data;
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

                            int8_t val = base_ptr[offset];
                            float val_fp32 = (float )val * scale;
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

                fprintf(pFile, "Shape is {%d %d}, data type is int8\n", height, width);

                int8_t * base_ptr = tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        int8_t val = base_ptr[offset];
                        float val_fp32 = (float )val * scale;
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

                fprintf(pFile, "Shape is {%d}, data type is int8\n", width);

                int8_t * base_ptr = tensor->data;

                for (int w = 0; w < width; w++)
                {
                    unsigned char val = base_ptr[w];
                    float val_fp32 = (float )val * scale;
                    if (val_fp32 < 0)
                        fprintf(pFile, "%.4f ", val_fp32);
                    else
                        fprintf(pFile, " %.4f ", val_fp32);
                }

                break;
            }
        }

        /* original int8 */
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

                fprintf(pFile, "Shape is {%d %d %d %d %d}, data type is int8, scale %f\n", dim5, batch, channel, height, width, scale);

                int8_t* base_ptr = tensor->data;

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

                                    int8_t val = base_ptr[offset];

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

                fprintf(pFile, "Shape is {%d %d %d %d}, data type is int8, scale %f\n", batch, channel, height, width, scale);

                int8_t* base_ptr = tensor->data;
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

                                int8_t val = base_ptr[offset];

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

                fprintf(pFile, "Shape is {%d %d %d}, data type is int8, scale %f\n", batch, channel, width, scale);

                int8_t* base_ptr = tensor->data;
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

                            int8_t val = base_ptr[offset];

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

                fprintf(pFile, "Shape is {%d %d}, data type is int8, scale %f\n", height, width, scale);

                int8_t* base_ptr = tensor->data;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int offset = 0;

                        offset += h * width;
                        offset += w;

                        int8_t val = base_ptr[offset];

                        fprintf(pFile, "%3d ", val);
                    }
                    fprintf(pFile, "\n");
                }

                break;
            }
            case 1: {
                int width = tensor->dims[0];

                fprintf(pFile, "Shape is {%d}, data type is int8, scale %f\n", width, scale);

                int8_t * base_ptr = tensor->data;

                for (int w = 0; w < width; w++)
                {
                    int8_t val = base_ptr[w];

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
#endif

#define INPLACE_BLOCK_FLAG 0x40
static void release_mem_pool(struct mem_pool* mem_pool);

struct mem_record
{
    struct ir_tensor* ir_tensor;
    int used;
    int block_id;
};

static int find_tensor_mem_list(struct vector* tensor_mem_list, const struct ir_tensor* ir_tensor)
{
    int rec_number = get_vector_num(tensor_mem_list);

    for (int i = 0; i < rec_number; i++)
    {
        struct mem_record* rec = ( struct mem_record* )get_vector_data(tensor_mem_list, i);

        if (rec->ir_tensor == ir_tensor)
            return i;
    }

    return -1;
}

static int init_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct ir_node* ir_node,
                          struct node_ops* node_ops)
{
    exec_node->ir_node = ir_node;
    exec_node->node_ops = node_ops;
    exec_node->ops_priv = NULL;
    exec_node->inplace_map_num = 0;
    exec_node->inplace_map_ptr = NULL;
    exec_node->shared_mem_size = 0;
    exec_node->shared_pack4_mem_size = 0;
    exec_node->output_num = ir_node->output_num;

    int8_t* block_id = exec_node->block_id;

    if (exec_node->output_num > 4)
    {
        exec_node->block_id_ptr = ( int8_t* )sys_malloc(sizeof(int8_t) * exec_node->output_num);
        block_id = exec_node->block_id_ptr;
    }

    for (int i = 0; i < exec_node->output_num; i++)
        block_id[i] = -1;

    if (node_ops->init_node && node_ops->init_node(node_ops, exec_node, exec_graph) < 0)
        return -1;

    return 0;
}

static void release_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node_ops* node_ops)
{
    if (node_ops->release_node)
        node_ops->release_node(node_ops, exec_node, exec_graph);

    if (exec_node->inplace_map_num > 2)
        sys_free(exec_node->inplace_map_ptr);

    if (exec_node->output_num > 4)
        sys_free(exec_node->block_id_ptr);
}

static struct exec_graph* new_exec_graph(void)
{
    struct exec_graph* exec_graph = ( struct exec_graph* )sys_malloc(sizeof(struct exec_graph));

    if (exec_graph == NULL)
        return NULL;

    exec_graph->exec_node_list = create_vector(sizeof(struct exec_node), NULL);

    if (exec_graph->exec_node_list == NULL)
    {
        sys_free(exec_graph);
        return NULL;
    }

    exec_graph->shared_mem = NULL;
    exec_graph->shared_mem_size = 0;
    exec_graph->mem_pool = NULL;

    exec_graph->shared_pack4_mem = NULL;
    exec_graph->shared_pack4_mem_size = 0;

    return exec_graph;
}

static void free_exec_graph_mem(struct exec_graph* graph)
{
    /* free the shared memory */
    if (graph->shared_mem)
    {
        sys_free(graph->shared_mem);
        graph->shared_mem = NULL;
        graph->shared_mem_size = 0;
    }
    /* free the shared pack4 memory */
    if (graph->shared_pack4_mem)
    {
        sys_free(graph->shared_pack4_mem);
        graph->shared_pack4_mem = NULL;
        graph->shared_pack4_mem_size = 0;
    }

    /* free the mem pool */
    if (graph->mem_pool)
    {
        release_mem_pool(graph->mem_pool);
        graph->mem_pool = NULL;
    }
}

static void release_exec_graph(void* exec_graph)
{
    struct exec_graph* graph = ( struct exec_graph* )exec_graph;

    int node_num = get_vector_num(graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        release_exec_node(graph, exec_node, node_ops);
    }

    free_exec_graph_mem(graph);

    release_vector(graph->exec_node_list);

    sys_free(graph);
}

static struct exec_graph* create_exec_graph(struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode)
{
    /* generate exec_graph */
    int node_num = subgraph->node_num;
    struct ir_graph* ir_graph = subgraph->graph;
    struct exec_graph* exec_graph = new_exec_graph();
    struct cpu_device* dev = ( struct cpu_device* )subgraph->nn_dev;

    if (exec_graph == NULL)
    {
        set_tengine_errno(ENOMEM);
        return NULL;
    }

    exec_graph->dev = dev;
    exec_graph->num_thread = num_thread;
    exec_graph->cpu_affinity = cpu_affinity;
    exec_graph->mode = mode;

    for (int i = 0; i < node_num; i++)
    {
        struct ir_node* ir_node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);

        // fprintf(stderr, "prerun: %d, %s\n", ir_node->op.op_type, ir_node->name);

        if (ir_node->op.op_type == OP_CONST || ir_node->op.op_type == OP_INPUT)
            continue;

        struct node_ops* node_ops = find_node_ops(exec_graph, ir_node);

        if (node_ops == NULL)
        {
            TLOG_ERR("%s: failed to find node ops for node: %d, %s\n", dev->base.name, ir_node->idx, ir_node->name);
            set_tengine_errno(EFAULT);
            goto error;
        }

        struct exec_node exec_node;

        if (init_exec_node(exec_graph, &exec_node, ir_node, node_ops) < 0)
        {
            TLOG_ERR("%s: failed to init exec node for node: %d, %s\n", dev->base.name, ir_node->idx, ir_node->name);
            set_tengine_errno(EFAULT);
            goto error;
        }

        push_vector_data(exec_graph->exec_node_list, &exec_node);
    }

    return exec_graph;

error:
    release_exec_graph(exec_graph);
    return NULL;
}

static int find_inplace_input(struct exec_node* exec_node, int output_slot, struct ir_node* ir_node,
                              struct ir_graph* ir_graph)
{
    if (exec_node->inplace_map_num == 0)
        return -1;

    uint8_t* inplace_map;

    if (exec_node->inplace_map_num > 2)
        inplace_map = exec_node->inplace_map_ptr;
    else
        inplace_map = exec_node->inplace_map;

    int i;
    for (i = 0; i < 2 * exec_node->inplace_map_num; i += 2)
    {
        if (inplace_map[i] == output_slot)
            break;
    }

    /* no map */
    if (i == 2 * exec_node->inplace_map_num)
        return -1;

    int input_slot = inplace_map[i + 1];

    struct ir_tensor* tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_slot]);

    if (tensor->consumer_num > 1)
        return -1;

    return input_slot;
}

static void mem_pool_dump(struct mem_pool* mem_pool)
{
    int block_number = get_vector_num(mem_pool->block_list);

    TLOG_INFO("block number: %d align size: %d\n", block_number, mem_pool->align_size);

    for (int i = 0; i < block_number; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        TLOG_INFO("%d: %p (%d) used: %d free: %d\n", i, entry->addr, entry->block_size, entry->alloc_count,
                  entry->free_count);
    }
}

static void* mem_pool_get_mem_block(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, block_id);

    unsigned long addr = ( long )(entry->addr);
    unsigned long aligned_addr = (addr + 4 + mem_pool->align_size) & (~(mem_pool->align_size - 1));

    return ( void* )aligned_addr;
}

static int mem_pool_get_backend_mem(struct mem_pool* mem_pool)
{
    int block_num = get_vector_num(mem_pool->block_list);

    for (int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        entry->block_size = entry->max_req_size + mem_pool->align_size + 128;

        entry->addr = sys_malloc(entry->block_size);

        if (entry->addr == NULL)
            return -1;
    }

    return 0;
}

static int mem_pool_allocate(struct mem_pool* mem_pool, int size)
{
    int block_num = get_vector_num(mem_pool->block_list);
    ;

    for (int i = 0; i < block_num; i++)
    {
        struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

        if (entry->free_count != entry->alloc_count)
            continue;

        /* TODO: use the best match alg */

        entry->alloc_count++;

        if (entry->max_req_size < size)
            entry->max_req_size = size;

        return i;
    }

    /* create new block */

    struct mem_block_entry e;

    e.addr = NULL;
    e.max_req_size = size;
    e.alloc_count = 1;
    e.free_count = 0;

    push_vector_data(mem_pool->block_list, &e);

    return block_num;
}

static void mem_pool_free(struct mem_pool* mem_pool, int block_id)
{
    struct mem_block_entry* block = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, block_id);

    block->free_count++;
}

static void release_mem_pool(struct mem_pool* mem_pool)
{
    if (mem_pool->block_list != NULL)
    {
        int block_num = get_vector_num(mem_pool->block_list);

        for (int i = 0; i < block_num; i++)
        {
            struct mem_block_entry* entry = ( struct mem_block_entry* )get_vector_data(mem_pool->block_list, i);

            sys_free(entry->addr);
        }

        release_vector(mem_pool->block_list);
    }

    sys_free(mem_pool);
}

static struct mem_pool* create_mem_pool(void)
{
    struct mem_pool* mem_pool = ( struct mem_pool* )sys_malloc(sizeof(struct mem_pool));

    if (mem_pool == NULL)
        return NULL;

    mem_pool->align_size = 16;
    mem_pool->block_list = create_vector(sizeof(struct mem_block_entry), NULL);

    if (mem_pool->block_list == NULL)
        goto error;

    mem_pool->allocate = mem_pool_allocate;
    mem_pool->free = mem_pool_free;
    mem_pool->dump = mem_pool_dump;
    mem_pool->get_backend_mem = mem_pool_get_backend_mem;
    mem_pool->get_mem_block = mem_pool_get_mem_block;

    return mem_pool;

error:

    release_mem_pool(mem_pool);

    return NULL;
}

static int alloc_exec_graph_mem(struct exec_graph* exec_graph)
{
    struct mem_pool* mem_pool;
    int max_shared_mem_size = 0;
    int max_shared_pack4_mem_size = 0;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    struct vector* tensor_mem_list = create_vector(sizeof(struct mem_record), NULL);

    if (tensor_mem_list == NULL)
        return -1;

    mem_pool = create_mem_pool();

    if (mem_pool == NULL)
        return -1;

    exec_graph->mem_pool = mem_pool;

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct ir_node* ir_node = exec_node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;

        int8_t* block_id;

        if (exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for (int j = 0; j < ir_node->output_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if (ir_tensor->data != NULL)
                continue;

            int inplace_input = find_inplace_input(exec_node, j, ir_node, ir_graph);

            if (inplace_input >= 0)
            {
                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[inplace_input]);

                int idx = find_tensor_mem_list(tensor_mem_list, input_tensor);

                /* if the input is from outside buffer, input_r should be NULL */
                if (idx < 0)
                    continue;

                struct mem_record* input_r = ( struct mem_record* )get_vector_data(tensor_mem_list, idx);

                input_r->ir_tensor = ir_tensor;
                input_r->used = ir_tensor->consumer_num;
                block_id[j] = INPLACE_BLOCK_FLAG | inplace_input;
                continue;
            }

            /* allocate mem from pool */
            int mem_size = ir_tensor->elem_size * ir_tensor->elem_num;

            struct mem_record r;

            r.ir_tensor = ir_tensor;
            r.block_id = mem_pool->allocate(mem_pool, mem_size);
            r.used = ir_tensor->consumer_num;

            block_id[j] = r.block_id;

            push_vector_data(tensor_mem_list, &r);
        }

        /* clear input tensor count */
        for (int j = 0; j < ir_node->input_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[j]);

            if (ir_tensor->data != NULL)
                continue;

            int idx = find_tensor_mem_list(tensor_mem_list, ir_tensor);

            if (idx < 0)
                continue;

            struct mem_record* input_r = ( struct mem_record* )get_vector_data(tensor_mem_list, idx);

            input_r->used--;

            if (input_r->used == 0)
            {
                mem_pool->free(mem_pool, input_r->block_id);
                remove_vector_by_idx(tensor_mem_list, idx);
            }
        }

        /* handle shared mem */
        if (exec_node->shared_mem_size > max_shared_mem_size)
            max_shared_mem_size = exec_node->shared_mem_size;
        if (exec_node->shared_pack4_mem_size > max_shared_pack4_mem_size)
            max_shared_pack4_mem_size = exec_node->shared_pack4_mem_size;
    }

    TLOG_DEBUG("final tensor_mem_list number: %d\n", get_vector_num(tensor_mem_list));

    release_vector(tensor_mem_list);

    exec_graph->shared_mem_size = max_shared_mem_size;
    exec_graph->shared_pack4_mem_size = max_shared_pack4_mem_size;

    if (max_shared_mem_size > 0)
    {
        exec_graph->shared_mem = sys_malloc(max_shared_mem_size);

        if (exec_graph->shared_mem == NULL)
        {
            TLOG_ERR("cannot allocate shared memory. size=%d\n", max_shared_mem_size);
            return -1;
        }
    }
    if (max_shared_pack4_mem_size > 0)
    {
        exec_graph->shared_pack4_mem = sys_malloc(max_shared_pack4_mem_size);

        if (exec_graph->shared_pack4_mem == NULL)
        {
            TLOG_ERR("cannot allocate shared pack4 memory. size=%d\n", max_shared_pack4_mem_size);
            return -1;
        }
    }

    TLOG_DEBUG("shared memory: %p size=%d\n", exec_graph->shared_mem, max_shared_mem_size);
    TLOG_DEBUG("shared pack4 memory: %p size=%d\n", exec_graph->shared_pack4_mem, max_shared_pack4_mem_size);

    if (mem_pool->get_backend_mem(mem_pool) < 0)
    {
        TLOG_ERR("cannot allocate enough memory from backend\n");
        return -1;
    }

    mem_pool->dump(mem_pool);

    /* now, the real allocate */
    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct ir_node* ir_node = exec_node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;
        struct mem_pool* mem_pool = exec_graph->mem_pool;

        int8_t* block_id;

        if (exec_node->output_num > 4)
            block_id = exec_node->block_id_ptr;
        else
            block_id = exec_node->block_id;

        for (int j = 0; j < ir_node->output_num; j++)
        {
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[j]);

            if (block_id[j] < 0)
                continue;

            if (block_id[j] & INPLACE_BLOCK_FLAG)
            {
                int input_idx = block_id[j] & (INPLACE_BLOCK_FLAG - 1);

                struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[input_idx]);
                ir_tensor->data = input_tensor->data;
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
            else
            {
                ir_tensor->data = mem_pool->get_mem_block(mem_pool, block_id[j]);
                ir_tensor->free_host_mem = 0;
                ir_tensor->internal_allocated = MEM_POOL_ALLOCATED;
            }
        }
    }

    return 0;
}

static int prerun_exec_graph(struct exec_graph* exec_graph)
{
    int node_num = get_vector_num(exec_graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* exec_node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = exec_node->node_ops;

        if (node_ops->prerun && node_ops->prerun(node_ops, exec_node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to prerun node %d\n", exec_graph->dev->base.name, exec_node->ir_node->idx);
            return -1;
        }
    }

    return 0;
}

static int prerun(struct nn_device* dev, struct subgraph* subgraph, int num_thread, int cpu_affinity, int mode)
{
    struct exec_graph* exec_graph;

    /* create exec_graph */
    exec_graph = create_exec_graph(subgraph, num_thread, cpu_affinity, mode);

    if (exec_graph == NULL)
        return -1;

    if (alloc_exec_graph_mem(exec_graph) < 0 || prerun_exec_graph(exec_graph) < 0)
    {
        release_exec_graph(exec_graph);
        return -1;
    }

    subgraph->exec_graph = exec_graph;

    return 0;
}
#ifdef DEBUG_TIME
static double get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
}
#endif

static int run(struct nn_device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = subgraph->exec_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        /* TODO: handle the shape changed  and dynamic shape case */
        if (node_ops->reshape && node_ops->reshape(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to reshape node %d, %s\n", dev->name, node->ir_node->idx, node->ir_node->name);
            return -1;
        }

        /* TODO: add dynamic skip feature */
#ifdef DEBUG_TIME
        double start = get_cur_time();
#endif
        if (node_ops->run(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to run node %d, %s\n", dev->name, node->ir_node->idx, node->ir_node->name);
            return -1;
        }
        char* name = node->ir_node->name;
#ifdef DEBUG_TIME
        double end = get_cur_time();
        fprintf(stderr, "%-20s  %8.2f ms  %s\n", get_op_name(node->ir_node->op.op_type), end - start, name);
#endif
#ifdef DEBUG_DATA
        struct ir_graph* ir_graph = node->ir_node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->input_tensors[0]);
        struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->output_tensors[0]);
        /* debug */
        if (input_tensor->dim_num <= 5)
            extract_feature_blob_f32("in", name, input_tensor);
        if (output_tensor->dim_num <= 5)
            extract_feature_blob_f32("out", name, output_tensor);
#endif

//#define DUMP_NODE_OUTPUT
#ifdef DUMP_NODE_OUTPUT
        /* dump the node output */
        struct ir_node* ir_node = node->ir_node;
        struct ir_graph* ir_graph = ir_node->graph;

        for (int i = 0; i < ir_node->input_num; i++)
        {
            char fname[128];
            struct ir_tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);

            sprintf(fname, "/tmp/dump/node%s%d.%d", (ir_node->idx < 10 ? "0" : ""), ir_node->idx, i);

            dump_float(fname, ir_tensor->data, ir_tensor->elem_num);
        }

#endif
    }

    return 0;
}

static int postrun(struct nn_device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = subgraph->exec_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* node = ( struct exec_node* )get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        if (node_ops->postrun && node_ops->postrun(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to postrun node %d\n", dev->name, node->ir_node->idx);
        }
    }

    release_exec_graph(exec_graph);

    subgraph->exec_graph = NULL;

    return 0;
}

static int cpu_dev_release_exec_graph(struct nn_device* dev, void* exec_graph)
{
    release_exec_graph(exec_graph);
    return 0;
}

static struct cpu_device cpu_dev = {
    .base = {.name = "cpu_dev",
             .prerun = prerun,
             .run = run,
             .postrun = postrun,
             .async_run = NULL,
             .async_wait = NULL,
             .release_exec_graph = cpu_dev_release_exec_graph,
             .init = NULL,
             .release = NULL},
    .master_cpu = 0,
    .cpu_model = 0,
};

int register_cpu_device(void)
{
    TLOG_INFO("Tengine plugin device %s is registered.\n", cpu_dev.base.name);
    return register_nn_device(&cpu_dev.base);
}

#ifndef STANDLONE_MODE
REGISTER_NN_DEVICE(&cpu_dev.base);
#endif
