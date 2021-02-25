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

#include "tengine_c_api.h"
#include "tengine_ir.h"
#include "tengine_op.h"
#include "tengine_exec.h"
#include "tengine_utils.h"
#include "vector.h"
#include "param_type.h"

const char* tensor_type_string(int tensor_type)
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

const char* data_type_string(int data_type)
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

int data_type_size(int data_type)
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

const char* layout_string(int layout)
{
    if (layout == TENGINE_LAYOUT_NHWC)
        return "NHWC";
    else
        return "NCHW";
}

const char* model_format_string(int model_format)
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

struct op_map_entry
{
    int op_type;
    const char* op_name;
};

static struct vector* op_map_list = NULL;

int register_op_map(int op_type, const char* name)
{
    if (NULL == op_map_list)
    {
        init_op_name_map();
    }

    struct op_map_entry e;

    e.op_type = op_type;
    e.op_name = name;

    return push_vector_data(op_map_list, &e);
}

int unregister_op_map(int op_type)
{
    int n = get_vector_num(op_map_list);
    int i;

    for (i = 0; i < n; i++)
    {
        struct op_map_entry* e = ( struct op_map_entry* )get_vector_data(op_map_list, i);

        if (e->op_type == op_type)
            break;
    }

    if (i == n)
        return -1;

    remove_vector_by_idx(op_map_list, i);

    return 0;
}

int init_op_name_map(void)
{
    if (NULL == op_map_list)
    {
        op_map_list = create_vector(sizeof(struct op_map_entry), NULL);

        if (op_map_list == NULL)
            return -1;
    }

    return 0;
}

void release_op_name_map(void)
{
    release_vector(op_map_list);
}

int get_op_type(const char* name)
{
    int map_num = get_vector_num(op_map_list);

    for (int i = 0; i < map_num; i++)
    {
        struct op_map_entry* e = get_vector_data(op_map_list, i);

        if (!strcmp(e->op_name, name))
            return e->op_type;
    }

    return -1;
}

const char* get_op_name(int op_type)
{
    int map_num = get_vector_num(op_map_list);

    for (int i = 0; i < map_num; i++)
    {
        struct op_map_entry* e = get_vector_data(op_map_list, i);

        if (e->op_type == op_type)
            return e->op_name;
    }

    return NULL;
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

int param_entry_type_mapping(const char* type_name)
{
    if (type_name == NULL)
        return PE_GENERIC;

    if (!strcmp(type_name, data_type_typeinfo_name(TENGINE_DT_INT32)))
        return PE_INT32;

    if (!strcmp(type_name, data_type_typeinfo_name(TENGINE_DT_FP32)))
        return PE_FP32;

    return PE_GENERIC;
}

void dump_float(const char* fname, float* data, int number)
{
    FILE* fp = fopen(fname, "w");

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
