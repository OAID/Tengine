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

#ifndef __TINY_GRAPH__
#define __TINY_GRAPH__

#include <stdint.h>

#define NN_TINY_VERSION_1 1
#define NN_OP_VERSION_1 1

#define NN_PAD_VALID 0
#define NN_PAD_SAME -1

#define MAX_TENSOR_DIM_NUM 4
#define MAX_NODE_INPUT_NUM 4

enum
{
    NN_OP_CONV,
    NN_OP_FC,
    NN_OP_POOL,
    NN_OP_RELU,
    NN_OP_SOFTMAX,
    NN_OP_MAX
};

enum
{
    NN_DT_Q7,
    NN_DT_Q15,
    NN_DT_Q31,
    NN_DT_FP32
};

enum
{
    NN_TENSOR_INPUT,
    NN_TENSOR_CONST,
    NN_TENSOR_VAR
};

enum
{
    NN_LAYOUT_NCHW,
    NN_LAYOUT_NHWC
};

enum
{
    NN_POOL_MAX,
    NN_POOL_AVG
};

struct tiny_tensor
{
    int dims[MAX_TENSOR_DIM_NUM];
    uint16_t shift;
    uint8_t dim_num;
    uint8_t data_type; /* Q7, Q15, FP32 */
    uint8_t tensor_type; /* input, const or variable */
    const void* data; /* Must be NULL for not const tensor */
};

struct tiny_node
{
    uint8_t input_num;
    uint8_t output_num;
    uint8_t op_type;
    /*
        operator version, decide how to interpret the content in op_param
   */
    uint8_t op_ver;
    const void* op_param;
    const struct tiny_tensor* input[MAX_NODE_INPUT_NUM];
    const struct tiny_tensor* output; /* assume one node just create one tensor */
};

struct tiny_graph
{
    char* name;
    uint8_t tiny_version; /* NN_TINY_VESION */
    uint8_t layout; /* NHWC or NCHW */
    uint8_t node_num;
    uint32_t nn_id;
    uint32_t create_time;
    const struct tiny_node** node_list;
};

/* op param definitions */

struct tiny_conv_param
{
    uint8_t kernel_h;
    uint8_t kernel_w;
    uint8_t stride_h;
    uint8_t stride_w;
    int8_t pad_h;
    int8_t pad_w;

    /* -1, no activation
        0, relu
        6, relu6
    */
    int8_t activation;
};

struct tiny_pool_param
{
    uint8_t pool_method;
    uint8_t kernel_h;
    uint8_t kernel_w;
    int8_t pad_h;
    int8_t pad_w;
    uint8_t stride_h;
    uint8_t stride_w;
};

extern const struct tiny_graph* get_tiny_graph(void);
extern void free_tiny_graph(const struct tiny_graph*);

#endif
