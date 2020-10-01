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
 * Author: bhu@openailab.com
 */

#include <math.h>
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "../../cpu_node_ops.h"
#include "tengine_op.h"
#include "eltwise_param.h"
#include <stdbool.h>

struct eltwise_op_param;

struct eltwise_op_param
{
    float scale[3];
    int zero[3];
};

#define ELT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ELT_MIN(a, b) ((a) < (b) ? (a) : (b))

static int ref_eltwise_fp32(void* output, void* input0, void* input1, int type, int input_count4, int input_chan,
                            int input_hw, int input1_count4, int num_thread, int input_hw_1, struct eltwise_param* eltwise_param)
{
    float* out_ptr = ( float* )output;
    float* in0 = ( float* )input0;
    float* in1 = ( float* )input1;

    switch (type)
    {
        case ELT_SUB:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) - in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) - (*in1++);
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] - in1[i / input_hw];
                }
            }
            else
                return -1;
            break;
        case ELT_SUM:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) + in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) + (*in1++);
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] + in1[i / input_hw];
                }
            }
            else if(input_hw == input_hw_1){
                for( int i = 0; i < input_chan; i++){
                    for(int j = 0; j < input_hw; j++){
                        *out_ptr++ = in0[i*input_hw + j] + in1[j];
                    }
                }
                // printf("%d %d \n", input1_count4, input_chan);
            }
            else
                return -1;
            break;
        case ELT_MAX:
            for (int i = 0; i < input_count4; ++i)
            {
                *out_ptr++ = ELT_MAX(in0[i], in1[i]);
            }
            break;
        case ELT_PROD:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = (*in0++) * in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] * in1[i];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *out_ptr++ = (in1[i]) * in0[0];
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] * in1[i / input_hw];
                }
            }
            else if (input_chan == input_count4){
                for(int i = 0; i < input1_count4; i++)
                {
                    *out_ptr++ = in0[i/input_hw] * in1[i];
                }
            }
            else
                return -1;
            break;
        case ELT_RSQRT:
            for (int i = 0; i < input_count4; ++i)
            {
                *out_ptr++ = 1 / sqrt(in0[i]);
            }
            break;
        case ELT_MIN_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                *out_ptr++ = ELT_MIN((*in0++), in1[0]);
            }
            break;
        case ELT_SUB_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                *out_ptr++ = (*in0++) - in1[0];
            }
            break;
        case ELT_PROD_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                *out_ptr++ = (*in0++) * in1[0];
            }
            break;
        case ELT_DIV:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] / in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    *out_ptr++ = in0[i] / in1[i];
                }
            }
            else if (input_count4 == 1)
            {
                for (int i = 0; i < input1_count4; ++i)
                {
                    *out_ptr++ = in0[0] / (*in1++);
                }
            }
            else
            {
                break;
            }
            break;
        case ELT_POW:
            if(input_count4 == 1){
                for(int i = 0; i < input1_count4; i++){
                    *out_ptr++ = pow(in0[0], in1[i]);
                }
            } else if (input1_count4 == 1){
                for(int i = 0; i < input1_count4; i++){
                    *out_ptr++ = pow(in0[0], in1[i]);
                }
            } else if (input_count4 == input1_count4){
                for(int i = 0; i < input_count4; i++){
                    *out_ptr++ = pow(in0[i], in1[i]);
                }
            } else {
                printf("Case not support \n");
            }
            break;
        case ELT_POWER:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = pow((eltwise_param->shift + eltwise_param->scale * in0[i]), eltwise_param->power);
            }
            break;
        case ELT_LOG:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = log(in0[i]);
            }
            break;
        case ELT_EXP:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = exp(in0[i]);
            }
            break;
        case ELT_SQRT:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = sqrt(in0[i]);
            }
            break;
        case ELT_FLOOR:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = floor(in0[i]);
            }
            break;
        case ELT_SQUARE:
            for(int i = 0; i < input_count4; i++){
                *out_ptr++ = pow(in0[i], 2);
            }
            break;
        default:
            break;
    }

    return 0;
}

static int ref_eltwise_uint8(struct ir_tensor* output_tensor, struct ir_tensor* input_tensor0,
                             struct ir_tensor* input_tensor1, int type, int input_count4, int input_chan, int input_hw,
                             int input1_count4, int num_thread, int input_hw_1, struct eltwise_param* eltwise_param)
{
    uint8_t* input0_uint8 = ( uint8_t* )input_tensor0->data;
    uint8_t* input1_uint8 = ( uint8_t* )input_tensor1->data;
    uint8_t* output_uint8 = ( uint8_t* )output_tensor->data;

    float in_scale0 = input_tensor0->scale;
    float in_scale1 = input_tensor1->scale;
    float out_scale = output_tensor->scale;
    int in_zero0 = input_tensor0->zero_point;
    int in_zero1 = input_tensor1->zero_point;
    int out_zero = output_tensor->zero_point;

    /* input dequant */
    float* in0 = ( float* )sys_malloc(input_tensor0->elem_num * sizeof(float));
    float* in1 = ( float* )sys_malloc(input_tensor1->elem_num * sizeof(float));
    float* out_ptr = ( float* )sys_malloc(output_tensor->elem_num * sizeof(float));

    for (int i = 0; i < input_tensor0->elem_num; i++)
        in0[i] = (input0_uint8[i] - in_zero0) * in_scale0;
    for (int i = 0; i < input_tensor1->elem_num; i++)
        in1[i] = (input1_uint8[i] - in_zero1) * in_scale1;

    /* eltwise operator */
    switch (type)
    {
        case ELT_SUB:
            if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] - in1[i];
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] - in1[i / input_hw];
                }
            }
            else if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] - in1[0];
                }
            }
            else
                return -1;
            break;
        case ELT_SUM:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] + in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] + in1[i];
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] + in1[i / input_hw];
                }
            }
            else if(input_hw == input_hw_1){
                for( int i = 0; i < input_chan; i++){
                    for(int j = 0; j < input_hw; j++){
                        out_ptr[i] = in0[i*input_hw + j] + in1[j];
                    }
                }
            }
            else
                return -1;
            break;
        case ELT_MAX:
            for (int i = 0; i < input_count4; ++i)
            {
                out_ptr[i] = ELT_MAX(in0[i], in1[i]);
            }
            break;
        case ELT_PROD:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] * in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] * in1[i];
                }
            }
            else if (input_chan == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] * in1[i / input_hw];
                }
            }
            else
                return -1;
            break;
        case ELT_RSQRT:
            for (int i = 0; i < input_count4; ++i)
            {
                out_ptr[i] = 1 / sqrt(in0[i]);
            }
            break;
        case ELT_MIN_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                out_ptr[i] = ELT_MIN(in0[i], in1[0]);
            }
            break;
        case ELT_SUB_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                out_ptr[i] = in0[i] - in1[0];
            }
            break;
        case ELT_PROD_SCALAR:
            for (int i = 0; i < input_count4; ++i)
            {
                out_ptr[i] = in0[i] * in1[0];
            }
            break;
        case ELT_DIV:
            if (input1_count4 == 1)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] / in1[0];
                }
            }
            else if (input_count4 == input1_count4)
            {
                for (int i = 0; i < input_count4; ++i)
                {
                    out_ptr[i] = in0[i] / in1[i];
                }
            }
            else if (input_count4 == 1)
            {
                for (int i = 0; i < input1_count4; ++i)
                {
                    out_ptr[i] = in0[0] / in1[i];
                }
            }
            else
            {
                break;
            }
            break;
        case ELT_POW:
            if(input_count4 == 1){
                for(int i = 0; i < input1_count4; i++){
                    out_ptr[i] = pow(in0[0], in1[i]);
                }
            } else if (input1_count4 == 1){
                for(int i = 0; i < input1_count4; i++){
                    out_ptr[i] = pow(in0[0], in1[i]);
                }
            } else if (input_count4 == input1_count4){
                for(int i = 0; i < input_count4; i++){
                    out_ptr[i] = pow(in0[i], in1[i]);
                }
            } else {
                printf("Case not support \n");
            }
            break;
        case ELT_POWER:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = pow((eltwise_param->shift + eltwise_param->scale * in0[i]), eltwise_param->power);
            }
            break;
        case ELT_LOG:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = log(in0[i]);
            }
            break;
        case ELT_EXP:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = exp(in0[i]);
            }
            break;
        case ELT_SQRT:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = sqrt(in0[i]);
            }
            break;
        case ELT_FLOOR:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = floor(in0[i]);
            }
            break;
        case ELT_SQUARE:
            for(int i = 0; i < input_count4; i++){
                out_ptr[i] = pow(in0[i], 2);
            }
            break;
        default:
            break;
    }


    /* output quant */
    for (int i = 0; i < output_tensor->elem_num; i++)
    {
        int output_data = round(out_ptr[i] / out_scale) + out_zero;
        output_uint8[i] = output_data; // adjust for QA Models test case(mobilenet_v2_1.0_quant_tfile.tmfile)
    }

    sys_free(in0);
    sys_free(in1);
    sys_free(out_ptr);

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
    struct ir_node* ir_node = exec_node->ir_node;
    struct ir_graph* ir_graph = ir_node->graph;
    struct ir_tensor* input_tensor0;
    struct ir_tensor* input_tensor1 = NULL;
    struct ir_tensor* output_tensor;

    input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct eltwise_param* eltwise_param = ( struct eltwise_param* )ir_node->op.param_mem;

    int layout = ir_graph->graph_layout;
    void* input0 = input_tensor0->data;
    void* input1 = NULL;
    void* output = output_tensor->data;
    int input1_count4 = 0;
    int input_hw_1 = 0;

    if (ir_node->input_num > 1)
    {
        input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        input1 = input_tensor1->data;
        input1_count4 = input_tensor1->elem_num;
        input_hw_1 = input_tensor1->dims[2]*input_tensor1->dims[3];
    }

    if (!input_tensor1 || input_tensor0->elem_num >= input_tensor1->elem_num)
    {
        int input_chan_0 = 0;
        int input_hw_0 = 0;
        int input0_count4 = input_tensor0->elem_num;

        if (layout == TENGINE_LAYOUT_NCHW)
        {
            input_chan_0 = input_tensor0->dims[1];
            input_hw_0 = input_tensor0->dims[2] * input_tensor0->dims[3];
        }
        else if (layout == TENGINE_LAYOUT_NHWC)
        {
            input_chan_0 = input_tensor0->dims[3];
            input_hw_0 = input_tensor0->dims[1] * input_tensor0->dims[2];
        }
        else
        {
            TLOG_ERR("unknown graph layout: %d\n", ir_graph->graph_layout);
            set_tengine_errno(EFAULT);
            return -1;
        }

        int ret = -1;
        if (input_tensor0->data_type == TENGINE_DT_FP32)
            ret = ref_eltwise_fp32(output, input0, input1, eltwise_param->type, input0_count4, input_chan_0, input_hw_0,
                                   input1_count4, exec_graph->num_thread, input_hw_1, eltwise_param);
        else
            ret = ref_eltwise_uint8(output_tensor, input_tensor0, input_tensor1, eltwise_param->type, input0_count4,
                                    input_chan_0, input_hw_0, input1_count4, exec_graph->num_thread, input_hw_1, eltwise_param);
        return ret;
    }
    else
    {
        int input_chan_0 = 0;
        int input_hw_0 = 0;
        int input0_count4 = input_tensor1->elem_num;
        input1_count4 = input_tensor0->elem_num;

        if (layout == TENGINE_LAYOUT_NCHW)
        {
            input_chan_0 = input_tensor1->dims[1];
            input_hw_0 = input_tensor1->dims[2] * input_tensor1->dims[3];
        }
        else if (layout == TENGINE_LAYOUT_NHWC)
        {
            input_chan_0 = input_tensor1->dims[3];
            input_hw_0 = input_tensor1->dims[1] * input_tensor1->dims[2];
        }
        else
        {
            TLOG_ERR("unknown graph layout: %d\n", ir_graph->graph_layout);
            set_tengine_errno(EFAULT);
            return -1;
        }

        int ret = -1;
        if (input_tensor1->data_type == TENGINE_DT_FP32)
            ret = ref_eltwise_fp32(output, input1, input0, eltwise_param->type, input0_count4, input_chan_0, input_hw_0,
                                   input1_count4, exec_graph->num_thread, input_hw_1, eltwise_param);
        else
            ret = ref_eltwise_uint8(output_tensor, input_tensor1, input_tensor0, eltwise_param->type, input0_count4,
                                    input_chan_0, input_hw_0, input1_count4, exec_graph->num_thread, input_hw_1, eltwise_param);
        return ret;
    }
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct ir_node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score};

static int reg_eltwise_hcl_ops(void* arg)
{
    return register_builtin_node_ops(OP_ELTWISE, &hcl_node_ops);
}

static int unreg_eltwise_hcl_ops(void* arg)
{
    return unregister_builtin_node_ops(OP_ELTWISE, &hcl_node_ops);
}

AUTO_REGISTER_OPS(reg_eltwise_hcl_ops);
AUTO_UNREGISTER_OPS(unreg_eltwise_hcl_ops);
