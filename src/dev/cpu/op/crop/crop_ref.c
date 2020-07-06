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
 * Author: chh@openailab.com
 */

#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "crop_param.h"
#include <math.h>

static int ref_crop_fp32(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor, struct crop_param* param,
                         int num_thread)
{
    float* input = input_tensor->data;
    float* output = output_tensor->data;

    int iDataC = input_tensor->dims[1];
    int iDataH = input_tensor->dims[2];
    int iDataW = input_tensor->dims[3];

    int oDataN = output_tensor->dims[0];
    int oDataC = output_tensor->dims[1];
    int oDataH = output_tensor->dims[2];
    int oDataW = output_tensor->dims[3];

    // MXNet
    if (param->flag == 1)
    {
        if (param->num_args == 1)
        {
            int offsetH = (iDataH - param->crop_h) / 2;
            int offsetW = (iDataW - param->crop_w) / 2;
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + offsetH;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + offsetW;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                    input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
        if (param->num_args == 2)
        {
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + param->offset_h;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + param->offset_w;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                    input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
    }
    // Caffe
    if (param->flag == 0)
    {
        if (param->axis == 1)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    int i_c = param->offset_c + c;
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                input[n * iDataC * iDataH * iDataW + i_c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
        }
        if (param->axis == 2)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor;
    struct ir_tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct crop_param* crop_param = ( struct crop_param* )ir_node->op.param_mem;

    ref_crop_fp32(input_tensor, output_tensor, crop_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_crop_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_CROP, &hcl_node_ops);
}

static int unreg_crop_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_CROP, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_crop_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_crop_hcl_ops);
