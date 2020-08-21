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
 * Author: jxyang@openailab.com
 */

#include <math.h>
#include <unistd.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "permute_param.h"

static void __hwc(const float* input, float* output, int hh, int ww, int cc, int wc, int hw)
{
    for (int h = 0; h < hh; ++h)
    {
        float* out_ptr = output + h * wc;

        for (int w = 0; w < ww; ++w)
        {
            for (int c = 0; c < cc; ++c)
            {
                const float* in_ptr = input + c * hw + h * ww;
                out_ptr[w * cc + c] = in_ptr[w];
            }
        }
    }
}

static void __chw(const float* input, float* output, int hh, int ww, int cc, int wc, int hw)
{
    for (int c = 0; c < cc; ++c)
    {
        float* output_ptr = output + c * hw;    // chw
        for (int h = 0; h < hh; ++h)
        {
            for (int w = 0; w < ww; ++w)
            {
                const float* input_ptr = input + h * wc + w * cc;    // input hwc + wc
                // hw + w = input_ptr[c]
                output_ptr[h * ww + w] = input_ptr[c];
            }
        }
    }
}

static void __hwc_u8(const uint8_t* input, uint8_t* output, int hh, int ww, int cc, int wc, int hw)
{
    for (int h = 0; h < hh; ++h)
    {
        uint8_t* out_ptr = output + h * wc;

        for (int w = 0; w < ww; ++w)
        {
            for (int c = 0; c < cc; ++c)
            {
                const uint8_t* in_ptr = input + c * hw + h * ww;
                out_ptr[w * cc + c] = in_ptr[w];
            }
        }
    }
}

static void __chw_u8(const uint8_t* input, uint8_t* output, int hh, int ww, int cc, int wc, int hw)
{
    for (int c = 0; c < cc; ++c)
    {
        uint8_t* output_ptr = output + c * hw;    // chw
        for (int h = 0; h < hh; ++h)
        {
            for (int w = 0; w < ww; ++w)
            {
                const uint8_t* input_ptr = input + h * wc + w * cc;    // input hwc + wc
                // hw + w = input_ptr[c]
                output_ptr[h * ww + w] = input_ptr[c];
            }
        }
    }
}

//static int ref_permute_fp32(const float* in_data, float* out_data, const permute_param_t* param, const int dims[], int layout)
static int ref_permute_fp32(const struct ir_tensor* input_tensor, const struct ir_tensor* output_tensor, const permute_param_t* param)
{
    float* in_data = input_tensor->data;
    float* out_data = output_tensor->data;
    const int* dims = input_tensor->dims;
    int layout = input_tensor->layout;

    int n;
    int c;
    int h;
    int w;
    if (layout == TENGINE_LAYOUT_NCHW)
    {
        n = dims[0];
        c = dims[1];
        h = dims[2];
        w = dims[3];
    }
    else
    {
        n = dims[0];
        h = dims[1];
        w = dims[2];
        c = dims[3];
    }

    int wc = w * c;
    int hw = h * w;
    int chw = c * hw;

    const float* input = in_data;
    float* output = out_data;
    if (param->order0 == 0 && param->order1 == 2 && param->order2 == 3 && param->order3 == 1)
    {
        for (int ii = 0; ii < n; ++ii)
        {
            __hwc(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if (param->order0 == 0 && param->order1 == 3 && param->order2 == 1 && param->order3 == 2)
    {
        for (int ii = 0; ii < n; ++ii)
        {
            __chw(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if ((param->order0 == 1) && (param->order1 == 0) && (param->order2 == 2))
    {
        int channel = dims[0];
        int width = dims[2];
        int height = dims[1];
        int _hw = height * width;
        int _cw = channel * width;
        for (int q = 0; q < height; q++)
        {
            float* outptr = output + q * _cw;

            for (int i = 0; i < channel; i++)
            {
                const float* ptr = input + i * _hw;

                for (int j = 0; j < width; j++)
                {
                    outptr[i * width + j] = ptr[q * width + j];
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int ref_permute_uint8(const struct ir_tensor* input_tensor, const struct ir_tensor* output_tensor, const permute_param_t* param)
{
    uint8_t* in_data = input_tensor->data;
    uint8_t* out_data = output_tensor->data;
    const int* dims = input_tensor->dims;
    int layout = input_tensor->layout;

    int n;
    int c;
    int h;
    int w;
    if (layout == TENGINE_LAYOUT_NCHW)
    {
        n = dims[0];
        c = dims[1];
        h = dims[2];
        w = dims[3];
    }
    else
    {
        n = dims[0];
        h = dims[1];
        w = dims[2];
        c = dims[3];
    }

    int wc = w * c;
    int hw = h * w;
    int chw = c * hw;

    const uint8_t* input = in_data;
    uint8_t* output = out_data;
    if (param->order0 == 0 && param->order1 == 2 && param->order2 == 3 && param->order3 == 1)
    {
        for (int ii = 0; ii < n; ++ii)
        {
            __hwc_u8(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if (param->order0 == 0 && param->order1 == 3 && param->order2 == 1 && param->order3 == 2)
    {
        for (int ii = 0; ii < n; ++ii)
        {
            __chw_u8(input, output, h, w, c, wc, hw);

            input += chw;
            output += chw;
        }
    }
    else if ((param->order0 == 1) && (param->order1 == 0) && (param->order2 == 2))
    {
        int channel = dims[0];
        int width = dims[2];
        int height = dims[1];
        int _hw = height * width;
        int _cw = channel * width;
        for (int q = 0; q < height; q++)
        {
            uint8_t* outptr = output + q * _cw;

            for (int i = 0; i < channel; i++)
            {
                const uint8_t* ptr = input + i * _hw;

                for (int j = 0; j < width; j++)
                {
                    outptr[i * width + j] = ptr[q * width + j];
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    permute_param_t* param = ( struct permute_param* )(ir_node->op.param_mem);
    int ret = -1;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_permute_fp32(input_tensor, output_tensor, param);
    else
        ret = ref_permute_uint8(input_tensor, output_tensor, param);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops permute_node_ops = {.prerun = NULL,
                                           .run = run,
                                           .reshape = NULL,
                                           .postrun = NULL,
                                           .init_node = init_node,
                                           .release_node = release_node,
                                           .score = score};

static int ret_permute_node_ops(void* arg)
{
    return register_builtin_node_ops(OP_PERMUTE, &permute_node_ops);
}

static int unret_permute_node_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_PERMUTE, &permute_node_ops);
}

AUTO_REGISTER_OPS(ret_permute_node_ops);
AUTO_UNREGISTER_OPS(unret_permute_node_ops);
