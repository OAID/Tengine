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
 * Author:
 */

#include "concat_kernel_ref.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

int ref_concat_int8(struct graph* ir_graph, struct node* ir_node, int axis)
{
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    float output_scale = output_tensor->scale;

    if (ir_node->input_num == 1)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

        int8_t* input_data = (int8_t*)input_tensor->data;
        int8_t* output_data = (int8_t*)output_tensor->data;

        for (int i = 0; i < input_tensor->elem_num; i++)
            output_data[i] = input_data[i];

        return 0;
    }

    int dims = output_tensor->dim_num;
    int positive_axis = axis < 0 ? dims + axis : axis;

    /* 1d */
    if (dims == 1)
    {
        int output_step = 0;
        for (int num = 0; num < ir_node->input_num; num++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

            float intput_scale = input_tensor->scale;
            float rescale = intput_scale / output_scale;

            int size = input_tensor->elem_num;

            int8_t* input_data = (int8_t*)input_tensor->data;
            int8_t* output_data = (int8_t*)output_tensor->data + output_step;

            for (int i = 0; i < size; i++)
            {
                int idata = roundf(input_data[i] * rescale);
                if (idata > 127)
                    idata = 127;
                else if (idata < -127)
                    idata = 127;
                output_data[i] = idata;
            }

            output_step += size;
        }
    }

    /* 2d */
    if (dims == 2 && positive_axis == 0)
    {
        int output_step = 0;
        for (int num = 0; num < ir_node->input_num; num++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

            float intput_scale = input_tensor->scale;
            float rescale = intput_scale / output_scale;

            int size = input_tensor->elem_num;

            int8_t* input_data = (int8_t*)input_tensor->data;
            int8_t* output_data = (int8_t*)output_tensor->data + output_step;

            for (int i = 0; i < size; i++)
            {
                int idata = roundf(input_data[i] * rescale);
                if (idata > 127)
                    idata = 127;
                else if (idata < -127)
                    idata = 127;
                output_data[i] = idata;
            }

            output_step += size;
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        int out_n = output_tensor->dims[0];
        int out_w = output_tensor->dims[1];

        for (int n = 0; n < output_tensor->dims[0]; n++)
        {
            int output_step = 0;
            for (int num = 0; num < ir_node->input_num; num++)
            {
                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                float intput_scale = input_tensor->scale;
                float rescale = intput_scale / output_scale;

                int in_n = input_tensor->dims[0];
                int in_w = input_tensor->dims[1];

                int8_t* input_data = (int8_t*)input_tensor->data + n * in_w;
                int8_t* output_data = (int8_t*)output_tensor->data + n * out_w + output_step;

                for (int i = 0; i < in_w; i++)
                {
                    int idata = roundf(input_data[i] * rescale);
                    if (idata > 127)
                        idata = 127;
                    else if (idata < -127)
                        idata = 127;
                    output_data[i] = idata;
                }

                output_step += in_w;
            }
        }
    }

    /* 3d */
    if (dims == 3 && positive_axis == 0)
    {
        int output_step = 0;
        for (int num = 0; num < ir_node->input_num; num++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

            float intput_scale = input_tensor->scale;
            float rescale = intput_scale / output_scale;

            int size = input_tensor->elem_num;

            int8_t* input_data = (int8_t*)input_tensor->data;
            int8_t* output_data = (int8_t*)output_tensor->data + output_step;

            for (int i = 0; i < size; i++)
            {
                int idata = roundf(input_data[i] * rescale);
                if (idata > 127)
                    idata = 127;
                else if (idata < -127)
                    idata = 127;
                output_data[i] = idata;
            }

            output_step += size;
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        int out_n = output_tensor->dims[0];
        int out_h = output_tensor->dims[1];
        int out_w = output_tensor->dims[2];
        int out_nstep = out_h * out_w;

        for (int n = 0; n < out_n; n++)
        {
            int output_step = 0;
            for (int num = 0; num < ir_node->input_num; num++)
            {
                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                float intput_scale = input_tensor->scale;
                float rescale = intput_scale / output_scale;

                int in_n = input_tensor->dims[0];
                int in_h = input_tensor->dims[1];
                int in_w = input_tensor->dims[2];
                int in_nstep = in_h * in_w;

                int8_t* input_data = (int8_t*)input_tensor->data + n * in_nstep;
                int8_t* output_data = (int8_t*)output_tensor->data + n * out_nstep + output_step;

                for (int i = 0; i < in_nstep; i++)
                {
                    int idata = roundf(input_data[i] * rescale);
                    if (idata > 127)
                        idata = 127;
                    else if (idata < -127)
                        idata = 127;
                    output_data[i] = idata;
                }

                output_step += in_nstep;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        int out_n = output_tensor->dims[0];
        int out_h = output_tensor->dims[1];
        int out_w = output_tensor->dims[2];
        int out_nstep = out_h * out_w;

        for (int n = 0; n < out_n; n++)
        {
            for (int h = 0; h < out_h; h++)
            {
                int output_step = 0;
                for (int num = 0; num < ir_node->input_num; num++)
                {
                    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                    float intput_scale = input_tensor->scale;
                    float rescale = intput_scale / output_scale;

                    int in_n = input_tensor->dims[0];
                    int in_h = input_tensor->dims[1];
                    int in_w = input_tensor->dims[2];
                    int in_nstep = in_h * in_w;

                    int8_t* input_data = (int8_t*)input_tensor->data + n * in_nstep + h * in_w;
                    int8_t* output_data = (int8_t*)output_tensor->data + n * out_nstep + h * out_w + output_step;

                    for (int i = 0; i < in_w; i++)
                    {
                        int idata = roundf(input_data[i] * rescale);
                        if (idata > 127)
                            idata = 127;
                        else if (idata < -127)
                            idata = 127;
                        output_data[i] = idata;
                    }

                    output_step += in_w;
                }
            }
        }
    }

    /* 4d */
    if (dims == 4 && positive_axis == 0)
    {
        int output_step = 0;
        for (int num = 0; num < ir_node->input_num; num++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

            float intput_scale = input_tensor->scale;
            float rescale = intput_scale / output_scale;

            int size = input_tensor->elem_num;

            int8_t* input_data = (int8_t*)input_tensor->data;
            int8_t* output_data = (int8_t*)output_tensor->data + output_step;

            for (int i = 0; i < size; i++)
            {
                int idata = roundf(input_data[i] * rescale);
                if (idata > 127)
                    idata = 127;
                else if (idata < -127)
                    idata = 127;
                output_data[i] = idata;
            }

            output_step += size;
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        int out_n = output_tensor->dims[0];
        int out_c = output_tensor->dims[1];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];
        int out_cstep = out_h * out_w;
        int out_nstep = out_c * out_cstep;

        for (int n = 0; n < out_n; n++)
        {
            int output_step = 0;
            for (int num = 0; num < ir_node->input_num; num++)
            {
                struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                float intput_scale = input_tensor->scale;
                float rescale = intput_scale / output_scale;

                int in_n = input_tensor->dims[0];
                int in_c = input_tensor->dims[1];
                int in_h = input_tensor->dims[2];
                int in_w = input_tensor->dims[3];
                int in_cstep = in_h * in_w;
                int in_nstep = in_c * in_cstep;

                int8_t* input_data = (int8_t*)input_tensor->data + n * in_nstep;
                int8_t* output_data = (int8_t*)output_tensor->data + n * out_nstep + output_step;

                for (int i = 0; i < in_nstep; i++)
                {
                    int idata = roundf(input_data[i] * rescale);
                    if (idata > 127)
                        idata = 127;
                    else if (idata < -127)
                        idata = 127;
                    output_data[i] = idata;
                }

                output_step += in_nstep;
            }
        }
    }

    if (dims == 4 && positive_axis == 2)
    {
        int out_n = output_tensor->dims[0];
        int out_c = output_tensor->dims[1];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];
        int out_cstep = out_h * out_w;
        int out_nstep = out_c * out_cstep;

        for (int n = 0; n < out_n; n++)
        {
            for (int c = 0; c < out_c; c++)
            {
                int output_step = 0;
                for (int num = 0; num < ir_node->input_num; num++)
                {
                    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                    float intput_scale = input_tensor->scale;
                    float rescale = intput_scale / output_scale;

                    int in_n = input_tensor->dims[0];
                    int in_c = input_tensor->dims[1];
                    int in_h = input_tensor->dims[2];
                    int in_w = input_tensor->dims[3];
                    int in_cstep = in_h * in_w;
                    int in_nstep = in_c * in_cstep;

                    int8_t* input_data = (int8_t*)input_tensor->data + n * in_nstep + c * in_cstep;
                    int8_t* output_data = (int8_t*)output_tensor->data + n * out_nstep + c * out_cstep + output_step;

                    for (int i = 0; i < in_cstep; i++)
                    {
                        int idata = roundf(input_data[i] * rescale);
                        if (idata > 127)
                            idata = 127;
                        else if (idata < -127)
                            idata = 127;
                        output_data[i] = idata;
                    }

                    output_step += in_cstep;
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        int out_n = output_tensor->dims[0];
        int out_c = output_tensor->dims[1];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];
        int out_cstep = out_h * out_w;
        int out_nstep = out_c * out_cstep;

        for (int n = 0; n < out_n; n++)
        {
            for (int c = 0; c < out_c; c++)
            {
                for (int h = 0; h < out_h; h++)
                {
                    int output_step = 0;
                    for (int num = 0; num < ir_node->input_num; num++)
                    {
                        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[num]);

                        float intput_scale = input_tensor->scale;
                        float rescale = intput_scale / output_scale;

                        int in_n = input_tensor->dims[0];
                        int in_c = input_tensor->dims[1];
                        int in_h = input_tensor->dims[2];
                        int in_w = input_tensor->dims[3];
                        int in_cstep = in_h * in_w;
                        int in_nstep = in_c * in_cstep;

                        int8_t* input_data = (int8_t*)input_tensor->data + n * in_nstep + c * in_cstep + h * in_w;
                        int8_t* output_data = (int8_t*)output_tensor->data + n * out_nstep + c * out_cstep + h * out_w + output_step;

                        for (int i = 0; i < in_w; i++)
                        {
                            int idata = roundf(input_data[i] * rescale);
                            if (idata > 127)
                                idata = 127;
                            else if (idata < -127)
                                idata = 127;
                            output_data[i] = idata;
                        }

                        output_step += in_w;
                    }
                }
            }
        }
    }

    return 0;
}
