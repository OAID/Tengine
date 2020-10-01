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
 * Author: qtang@openailab.com
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "interp_param.h"

#define INTERP_MIN(a, b) ((a) < (b) ? (a) : (b))

void linear_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = ( double )w / outw;

    for(int dx = 0; dx < outw; dx++)
    {
        float fx = ( float )((dx + 0.5) * scale - 0.5);
        int sx = floor(fx);
        fx -= sx;

        if(sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if(sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

void resize_bilinear_image(float* src, float* dst, float* alpha, int* xofs, float* beta, int* yofs, int out_h, int out_w, int in_h, int in_w)
{
    int w = out_w;  //dst.w;
    int h = out_h;  //dst.h;

    // loop body
    float* rowsbuf0 = ( float* )sys_malloc(w * sizeof(float));
    float* rowsbuf1 = ( float* )sys_malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    memset(rowsbuf0, 0, w * sizeof(float));
    memset(rowsbuf1, 0, w * sizeof(float));

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
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src + (sy+1)*in_w;   //src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src + sy*in_w;       //src.row(sy);
            const float* S1 = src + (sy+1)*in_w;   //src.row(sy+1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0]*a0 + S0p[1]*a1;
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }

        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst + dy * out_w; //dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float temp = *rows0p++ * b0 + *rows1p++ * b1;
            *Dp++ = temp;
        }

        beta += 2;
    }

    sys_free(rowsbuf0);
    sys_free(rowsbuf1);
}

int ref_interp_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct interp_param* param)
{
    float* input = input_tensor->data;
    float* output = output_tensor->data;

    int batch = input_tensor->dims[0];
    int channel = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];

    int in_channel_size = in_h * in_w;
    int out_channel_size = out_h * out_w;

    int* buf = sys_malloc((param->output_width + param->output_height + param->output_width*2 + param->output_height*2)*sizeof(float));

    if (buf == NULL)
    {
        printf("interp malloc failed!\n");
        return -1;
    }

    int* xofs = buf;//new int[ow];
    int* yofs = buf + param->output_width ;//new int[oh];

    float* alpha = (float*)(buf + param->output_width  + param->output_height);//new float[ow * 2];
    float* beta = (float*)(buf + param->output_width + param->output_height + param->output_width*2);//new float[oh * 2];

    linear_coeffs(in_w, out_w, xofs, alpha);
    linear_coeffs(in_h, out_h, yofs, beta);

    for (int q = 0; q < channel; ++q)
    {
        resize_bilinear_image(input+in_channel_size*q, output+out_channel_size*q, alpha, xofs, beta, yofs, out_h, out_w, in_h, in_w);
    }

    sys_free(buf);

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

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;
    struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct interp_param* param = ( struct interp_param* )node->op.param_mem;

    int ret = ref_interp_fp32(input_tensor, output_tensor, param);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_relu_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_INTERP, &hcl_node_ops);
}

static int unreg_relu_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_INTERP, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_relu_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_relu_hcl_ops);
