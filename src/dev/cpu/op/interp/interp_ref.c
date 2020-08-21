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

int ref_interp_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct interp_param* param)
{
    if (input_tensor->dim_num != 4)
    {
        printf("interp dim num is not 4\n");
        return -1;
    }

    float* input = input_tensor->data;
    float* output = output_tensor->data;

    int batch = input_tensor->dims[0];
    int channel = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];    

    for (int n = 0; n < batch; ++n) 
    {
        for (int c = 0; c < channel; ++c) 
        {
            for (int y = 0; y < param->output_height; ++y) 
            {
                float in_y = INTERP_MIN(y / param->height_scale, (float)(in_h - 1));
                const int in_y1 = INTERP_MIN((int)(in_y), in_h - 1);
                const int in_y2 = INTERP_MIN(in_y1 + 1, in_h - 1);
                float dy1 = fabs(in_y - in_y1);
                float dy2 = fabs(in_y - in_y2);
                if (in_y1 == in_y2) 
                {
                    dy1 = 0.5f;
                    dy2 = 0.5f;
                }

                const int input_width_mul_y1 = in_w * in_y1;
                const int input_width_mul_y2 = in_w * in_y2;

                for (int x = 0; x < param->output_width; ++x) 
                {
                    float in_x = INTERP_MIN(x / param->width_scale, (float)(in_w - 1));
                    const int in_x1 = INTERP_MIN((int)(in_x), in_w - 1);
                    const int in_x2 = INTERP_MIN(in_x1 + 1, in_w - 1);

                    float dx1 = fabs(in_x - in_x1);
                    float dx2 = fabs(in_x - in_x2);
                    if (in_x1 == in_x2) 
                    {
                        dx1 = 0.5f;
                        dx2 = 0.5f;
                    }

                    float X11 = input[input_width_mul_y1 + in_x1];
                    float X21 = input[input_width_mul_y1 + in_x2];
                    float X12 = input[input_width_mul_y2 + in_x1];
                    float X22 = input[input_width_mul_y2 + in_x2];
                    output[param->output_width * y + x] = dx2 * dy2 * X11 +dx1 * dy2 * X21 +dx2 * dy1 * X12 +dx1 * dy1 * X22;
                }
            }
            input += in_h * in_w;
            output += param->output_width * param->output_height;
        }
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
