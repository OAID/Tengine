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
 * Author: sqfu@openailab.com
 * Author: qtang@openailab.com (update 20200611)
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "pad_param.h"

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static void pad(float* input, float* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v)
{
    float* ptr = input;
    float* outptr = output;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
    // fill center
    for (; y < (top + in_h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (in_w < 12)
        {
            for (; x < (left + in_w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            //            memcpy(outptr + left, ptr, in_w * sizeof(float));
            //            x += in_w;
            for (; x < in_w; x++)
            {
                outptr[left + x] = ptr[x];
            }
        }
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        ptr += in_w;
        outptr += out_w;
    }
    // fill bottom
    for (; y < out_h; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pad_param* param = ( struct pad_param* )ir_node->op.param_mem;

    int batch = input_tensor->dims[0];
    int channel = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int in_cstep = in_h * in_w;
    int in_size = channel * in_h * in_w;

    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_cstep = out_h * out_w;
    int out_size = channel * out_h * out_w;

    int pad_top = param->pad_2_h;
    int pad_bottom = param->pad_2_w;
    int pad_left = param->pad_3_h;
    int pad_right = param->pad_3_w;

    if (param->mode == 0)
    {
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channel; c++)
            {
                float* input_data = ( float* )input_tensor->data + n * in_size + c * in_cstep;
                float* output_data = ( float* )output_tensor->data + n * out_size + c * out_cstep;

                pad(input_data, output_data, in_h, in_w, out_h, out_w, pad_top, pad_left, param->value);
            }
        }
    }
    else
    {
        fprintf(stderr, "another mode dose not support, pad mode value %d\n", param->mode);
        return -1;
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops pad_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_pad_ops(void* arg)
{
    return register_builtin_node_ops(OP_PAD, &pad_node_ops);
}

static int unreg_pad_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_PAD, &pad_node_ops);
}

AUTO_REGISTER_OPS(reg_pad_ops);
AUTO_UNREGISTER_OPS(unreg_pad_ops);
