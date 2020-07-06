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
 * Author: zpluo@openailab.com
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

typedef struct __interp_param
{
    float width_scale;
    float height_scale;
    int batch_number;
    int inc;
    int inh;
    int inw;
    int output_width;
    int output_height;
    int in_channel_size;
    int out_channel_size;
    int* buf;

} _interp_param, *p_interp_param;

void linear_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = ( double )w / outw;

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = ( float )((dx + 0.5) * scale - 0.5);
        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

void resize_bilinear_image(float* src, float* dst, float* alpha, int* xofs, float* beta, int* yofs, int out_h,
                           int out_w, int in_h, int in_w)
{
    int w = out_w;    // dst.w;
    int h = out_h;    // dst.h;

    // loop body
    float* rowsbuf0 = ( float* )sys_malloc(w * sizeof(float));
    float* rowsbuf1 = ( float* )sys_malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
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
            const float* S1 = src + (sy + 1) * in_w;    // src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src + sy * in_w;    // src.row(sy);
            const float* S1 = src + (sy + 1) * in_w;    // src.row(sy+1);

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
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst + dy * out_w;    // dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }

    sys_free(rowsbuf0);
    sys_free(rowsbuf1);
    rows0 = NULL;
    rows1 = NULL;
}

int ref_interp_fp32(float* input, float* output, p_interp_param param)
{
    int* xofs = param->buf;    // new int[ow];
    int* yofs = param->buf + param->output_width;    // new int[oh];

    float* alpha = ( float* )(param->buf + param->output_width + param->output_height);    // new float[ow * 2];
    float* beta = ( float* )(param->buf + param->output_width + param->output_height +
                             param->output_width * 2);    // new float[oh * 2];

    linear_coeffs(param->inw, param->output_width, xofs, alpha);
    linear_coeffs(param->inh, param->output_height, yofs, beta);

    for (int q = 0; q < param->inc; ++q)
    {
        resize_bilinear_image(input + param->in_channel_size * q, output + param->out_channel_size * q, alpha, xofs,
                              beta, yofs, param->output_height, param->output_width, param->inh, param->inw);
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* node = exec_node->ir_node;
    struct ir_graph* graph = node->graph;

    struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct ir_tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    void* input = input_tensor->data;
    void* output = output_tensor->data;

    struct interp_param* param = ( struct interp_param* )node->op.param_mem;

    float width_scale = param->width_scale;
    float height_scale = param->height_scale;

    int batch_number = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];

    int output_width = inh * width_scale;
    int output_height = inw * height_scale;

    _interp_param op_param;

    op_param.inc = inc;
    op_param.inh = inh;
    op_param.inw = inw;
    op_param.batch_number = batch_number;
    op_param.output_width = output_width;
    op_param.output_height = output_height;
    op_param.width_scale = width_scale;
    op_param.height_scale = height_scale;
    op_param.out_channel_size = output_height * output_width;
    op_param.in_channel_size = inh * inw;

    op_param.buf = ( int* )malloc(sizeof(int) * (param->output_width + param->output_height + param->output_width * 2 +
                                                 param->output_height * 2));
    int ret = ref_interp_fp32(input, output, &op_param);
    free(op_param.buf);

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
