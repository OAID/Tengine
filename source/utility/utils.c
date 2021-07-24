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
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#include "utility/utils.h"

#include "defines.h"
#include "api/c_api.h"
#include "operator/op.h"
#include "module/module.h"
#include "utility/vector.h"

#include <stdio.h>
#include <string.h>

const char* get_tensor_type_string(int tensor_type)
{
    switch (tensor_type)
    {
    case TENSOR_TYPE_VAR:
        return "var";
    case TENSOR_TYPE_CONST:
        return "const";
    case TENSOR_TYPE_INPUT:
        return "input";
    case TENSOR_TYPE_DEP:
        return "dep";
    default:
        return "unknown";
    }
}

const char* get_tensor_layout_string(int layout)
{
    if (layout == TENGINE_LAYOUT_NHWC)
        return "NHWC";
    else
        return "NCHW";
}

const char* get_model_format_string(int model_format)
{
    switch (model_format)
    {
    case MODEL_FORMAT_TENGINE:
        return "tengine";
    case MODEL_FORMAT_CAFFE:
        return "caffe";
    case MODEL_FORMAT_ONNX:
        return "onnx";
    case MODEL_FORMAT_MXNET:
        return "mxnet";
    case MODEL_FORMAT_TENSORFLOW:
        return "tensorflow";
    case MODEL_FORMAT_TFLITE:
        return "tflite";
    case MODEL_FORMAT_DLA:
        return "dla";
    default:
        return "unknown";
    }
}

int get_op_type_from_name(const char* name)
{
    int count = get_op_method_count();
    for (int i = 0; i < count; i++)
    {
        ir_method_t* method = find_op_method_via_index(i);
        const char* op_name = find_op_name(method->type);

        if (0 == strcmp(op_name, name))
        {
            return method->type;
        }
    }

    return -1;
}

const char* get_op_name_from_type(int op_type)
{
    return find_op_name(op_type);
}

int get_tenser_element_size(int data_type)
{
    switch (data_type)
    {
    case TENGINE_DT_FP32:
    case TENGINE_DT_INT32:
        return 4;
    case TENGINE_DT_FP16:
    case TENGINE_DT_INT16:
        return 2;
    case TENGINE_DT_INT8:
    case TENGINE_DT_UINT8:
        return 1;
    default:
        return 0;
    }
}

const char* get_tensor_data_type_string(int data_type)
{
    switch (data_type)
    {
    case TENGINE_DT_FP32:
        return "fp32";
    case TENGINE_DT_FP16:
        return "fp16";
    case TENGINE_DT_INT8:
        return "int8";
    case TENGINE_DT_UINT8:
        return "uint8";
    case TENGINE_DT_INT32:
        return "int32";
    case TENGINE_DT_INT16:
        return "int16";
    default:
        return "unknown";
    }
}

const char* data_type_typeinfo_name(int data_type)
{
    switch (data_type)
    {
    case TENGINE_DT_INT32:
        return "i";
    case TENGINE_DT_FP32:
        return "f";
    default:
        return NULL;
    }
}

void dump_float(const char* file_name, float* data, int number)
{
    FILE* fp = fopen(file_name, "w");

    for (int i = 0; i < number; i++)
    {
        if (i % 16 == 0)
        {
            fprintf(fp, "\n%d:", i);
        }
        fprintf(fp, " %.5g", data[i]);
    }

    fprintf(fp, "\n");

    fclose(fp);
}

int get_mask_count(size_t mask)
{
    int count = 0;

    for (int i = 0; i < sizeof(mask) * 8; i++)
        if (mask & (1 << i))
            count++;

    return count;
}

int get_mask_index(size_t mask)
{
    if (get_mask_count(mask) > 1)
    {
        return -1;
    }

    for (int i = 0; i < sizeof(mask) * 8; i++)
    {
        if (mask & (1 << i))
        {
            return i;
        }
    }

    return 0;
}
