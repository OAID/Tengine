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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/resize.hpp"
#include <math.h>

namespace TEngine {

namespace ResizeImpl {

struct ResizeOps : public MTNodeOps
{
    void bilinear_resize(float* inp, float* output, int h, int w, int c, float scale_x, float scale_y, int oh, int ow)
    {
        int out_hw = oh * ow;
        int in_hw = h * w;
        for(int j = 0; j < oh; j++)
        {
            float fy = (j + 0.5) * scale_y - 0.5;
            int sy = floor(fy);
            fy -= sy;
            sy = std::min(sy, h - 2);
            sy = std::max(0, sy);
            float fy_0 = 1.f - fy;

            for(int i = 0; i < ow; i++)
            {
                float fx = (i + 0.5) * scale_x - 0.5;
                int sx = floor(fx);
                fx -= sx;
                if(sx < 0)
                {
                    sx = 0;
                    fx = 0;
                }
                if(sx >= w - 1)
                {
                    fx = 0;
                    sx = w - 2;
                }
                float fx_0 = 1.f - fx;
                int out_idx = j * ow + i;
                int in_idx = sy * w + sx;
                // printf("i=%d j=%d\t sx=%d fx=%f\t sy=%d fy=%f\n",i,j,sx,fx,sy,fy);
                for(int k = 0; k < c; k++)
                {
                    int in_index = in_idx + k * in_hw;
                    output[k * out_hw + out_idx] = inp[in_index] * fx_0 * fy_0 + inp[in_index + w] * fx_0 * fy +
                                                   inp[in_index + 1] * fx * fy_0 + inp[in_index + w + 1] * fx * fy;
                }
            }
        }
    }

    void bilinear_resize_2x2(float* input, float* out, int h, int w, int c, int oh, int ow)
    {
        int out_hw = oh * ow;
        int in_hw = h * w;

        float* output = out;
        float* inp = input;
        for(int k = 0; k < c; k++)
        {
            // j=0
            {
                int sx = 0;
                output[0] = inp[0] * 0.25 + inp[w] * 0.75;
                for(int i = 1; i < ow - 1; i += 2)
                {
                    output[i] =
                        inp[sx] * 0.1875 + inp[sx + w] * 0.5625 + inp[sx + 1] * 0.0625 + inp[sx + w + 1] * 0.1875;

                    output[i + 1] =
                        inp[sx] * 0.0625 + inp[sx + w] * 0.1875 + inp[sx + 1] * 0.1875 + inp[sx + w + 1] * 0.5625;
                    sx += 1;
                }
                output[ow - 1] = inp[sx - 1] * 0.25 + inp[sx - 1 + w] * 0.75;
            }
            int sy = 0;
            for(int j = 1; j < oh - 1; j += 2)
            {
                // i=0;
                int sx = 0;
                {
                    int out_idx = j * ow;
                    int in_index = sy * w + sx;

                    output[out_idx] = inp[in_index] * 0.75 + inp[in_index + w] * 0.25;
                    output[out_idx + ow] = inp[in_index] * 0.25 + inp[in_index + w] * 0.75;
                }
                for(int i = 1; i < ow - 1; i += 2)
                {
                    int out_idx = j * ow + i;
                    int in_index = sy * w + sx;
                    float temp23 = (inp[in_index + w] + inp[in_index + 1]) * 0.1875;
                    float temp14 = (inp[in_index] + inp[in_index + w + 1]) * 0.1875;
                    output[out_idx] = inp[in_index] * 0.5625 + temp23 + inp[in_index + w + 1] * 0.0625;
                    output[out_idx + ow] = temp14 + inp[in_index + w] * 0.5625 + inp[in_index + 1] * 0.0625;
                    output[out_idx + 1] = temp14 + inp[in_index + w] * 0.0625 + inp[in_index + 1] * 0.5625;
                    output[out_idx + 1 + ow] = inp[in_index] * 0.0625 + temp23 + inp[in_index + w + 1] * 0.5625;
                    sx += 1;
                }
                //
                {
                    int i = ow - 1;
                    sx -= 1;

                    int out_idx = j * ow + i;
                    int in_index = sy * w + sx;

                    output[out_idx] = inp[in_index] * 0.75 + inp[in_index + w] * 0.25;
                    output[out_idx + ow] = inp[in_index] * 0.25 + inp[in_index + w] * 0.75;
                }
                sy += 1;
            }

            {
                int j_ow = (oh - 1) * ow;

                sy -= 1;

                // i=0;
                int sx = 0;
                {
                    int in_index = sy * w + sx;
                    output[j_ow] = inp[in_index] * 0.75 + inp[in_index + w] * 0.25;
                }
                for(int i = 1; i < ow - 1; i += 2)
                {
                    int in_index = sy * w + sx;

                    output[j_ow + i] = inp[in_index] * 0.5625 + (inp[in_index + w] + inp[in_index + 1]) * 0.1875 +
                                       inp[in_index + w + 1] * 0.0625;

                    output[j_ow + i + 1] = (inp[in_index] + inp[in_index + w + 1]) * 0.1875 +
                                           inp[in_index + w] * 0.0625 + inp[in_index + 1] * 0.5625;
                    sx += 1;
                }
                //
                {
                    int i = ow - 1;
                    sx -= 1;
                    int in_index = sy * w + sx;

                    output[j_ow + i] = inp[in_index] * 0.75 + inp[in_index + w] * 0.25;
                }
            }
            output += out_hw;
            inp += in_hw;
        }
    }

    void nearest_neighbor_resize(float* inp, float* out, int h, int w, int c_start, int c_end, float scale_x,
                                 float scale_y, int oh, int ow)
    {
        float* output;
        float* input;
        int si, sj;
        for(int k = c_start; k < c_end; k++)
        {
            input = inp + k * h * w;
            output = out + k * oh * ow;
            for(int i = 0; i < oh; i++)
            {
                si = std::min(( int )(i * scale_y), h - 1);
                for(int j = 0; j < ow; j++)
                {
                    sj = std::min(( int )(j * scale_x), w - 1);
                    output[i * ow + j] = input[si * w + sj];
                }
            }
        }
    }

    struct resize_param
    {
        float* input;
        float* output;
        int in_h;
        int in_w;
        int c_start;
        int c_end;
        int out_h;
        int out_w;
        float scale_x;
        float scale_y;
    };

    bool resize_aider(int cpu, int seq, void* data)
    {
        resize_param* param = ( resize_param* )(data);
        nearest_neighbor_resize(param->input, param->output, param->in_h, param->in_w, param->c_start, param->c_end,
                                param->scale_x, param->scale_y, param->out_h, param->out_w);
        return true;
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        Resize* resize_op = dynamic_cast<Resize*>(node->GetOp());
        ResizeParam* param_ = resize_op->GetParam();

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int batch_number = dims[0];

        int in_chw = dims[1] * dims[2] * dims[3];

        const TShape& shape1 = output_tensor->GetShape();
        const std::vector<int> out_dims = shape1.GetDim();

        int out_chw = out_dims[1] * out_dims[2] * out_dims[3];

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float scale_x = 1.f / param_->scale_w;
        float scale_y = 1.f / param_->scale_h;
        int cpu_number = cpu_info->GetCPUNumber();

        if(param_->type == 0)
        {
            for(int i = 0; i < batch_number; i++)
            {
                if(cpu_number == 1)
                {
                    nearest_neighbor_resize(input, output, dims[2], dims[3], 0, dims[1], scale_x, scale_y, out_dims[2],
                                            out_dims[3]);
                    input += in_chw;
                    output += out_chw;
                }
                else
                {
                    std::vector<sub_op_task> task_list;
                    std::vector<resize_param> param_list;
                    int steps = dims[1] / cpu_number;
                    param_list.resize(cpu_number);

                    auto f = std::bind(&ResizeOps::resize_aider, this, std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3);
                    for(int i = 0; i < cpu_number; i++)
                    {
                        sub_op_task tmp_task;
                        resize_param* param0 = &param_list[task_list.size()];
                        sub_op_task* task = &tmp_task;
                        task->exec_func = f;
                        task->seq = i;
                        task->data = param0;

                        param0->input = input;
                        param0->output = output;
                        param0->in_h = dims[2];
                        param0->in_w = dims[3];
                        param0->c_start = i * steps;
                        param0->c_end = param0->c_start + steps;
                        param0->out_h = out_dims[2];
                        param0->out_w = out_dims[3];
                        param0->scale_x = scale_x;
                        param0->scale_y = scale_y;
                        task_list.emplace_back(tmp_task);
                    }
                    task_dispatch(task_list, -1);
                    wait_done();
                }
            }
        }
        else
        {
            for(int i = 0; i < batch_number; i++)
            {
                bilinear_resize(input, output, dims[2], dims[3], dims[1], scale_x, scale_y, out_dims[2], out_dims[3]);
                input += in_chw;
                output += out_chw;
            }
        }

        return true;
    }
};

}    // namespace ResizeImpl

using namespace ResizeImpl;

void RegisterResizeNodeExec(void)
{
    ResizeOps* ops = new ResizeOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Resize", ops);
}

}    // namespace TEngine