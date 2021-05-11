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
 * Author: zpeng@openailab.com
 */

// TODO: this kernel is unbearable, needs to be refactor next version. @lswang

#include "utility/sys_port.h"

#include <string.h>
#include <math.h>
#include <stdio.h>


#define FLOAT_MAX 3.4028235E38
#define FLOAT_MIN -3.4028235E38

void sum_5d_ax1(int* dims, int dim_num, float* data, float* tmp);

void sum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void sum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void sum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void sum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void sum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void mean_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void mean_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void mean_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void mean_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void mean_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void mean_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void mean_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void mean_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void mean_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void asum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void asum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void asum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void asum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void asum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void asum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void asum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void asum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void asum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void sqsum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sqsum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sqsum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sqsum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sqsum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void sqsum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void sqsum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void sqsum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void sqsum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void max_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void max_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void max_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void max_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void max_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void max_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void max_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void max_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void max_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void min_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void min_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void min_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void min_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void min_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void min_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void min_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void min_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void min_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void prod_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void prod_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void prod_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void prod_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void prod_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void prod_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void prod_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void prod_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void prod_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void l2_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void l2_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void l2_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void l2_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void l2_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void l2_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void l2_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void l2_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void l2_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void logsum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void logsum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void logsum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void logsum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0);
void logsum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1);

void logsumexp_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsumexp_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsumexp_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void logsumexp_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sumexp_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sumexp_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sumexp_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);
void sumexp_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp);

struct reduce_param_ref
{
    int layout;
    int type;
    int param_dim[4];
};

static int ref_reduce_fp32(float* data, float* out_data, int dim0, int dim1, int dim2, int dim3, int out_size,
                           struct reduce_param_ref* param, int dim_num, int* dims)
{
    int offset = 0;
    float* tmp = ( float* )sys_malloc(sizeof(float) * out_size);
    memset(tmp, 0, sizeof(float) * out_size);
    int param_dim0 = param->param_dim[0];
    int param_dim1 = param->param_dim[1];
    int param_dim2 = param->param_dim[2];
    int param_dim3 = param->param_dim[3];

    if (param->type == 0)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            tmp[0] += data[offset];
                        }
                    }
                }
            }
        }
        else if(param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2 && (dim_num > 4))
        {
            if(dim_num == 5){
                sum_5d_ax1(dims, dim_num, data, tmp);
            }
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2 && (dim_num <= 4) )
        {
            fprintf(stderr, "wrond dim_num %d \n", dim_num);
            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            sum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            sum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            sum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            sum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce mean
    else if (param->type == 1)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp / (dim0 * dim1 * dim2 * dim3);
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            mean_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            mean_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            mean_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            mean_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            mean_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            mean_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            mean_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            mean_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            mean_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            mean_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            mean_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            mean_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            mean_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            mean_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            mean_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            mean_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            mean_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            mean_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            mean_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce asum
    else if (param->type == 2)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += fabs(data[offset]);
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            sum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            sum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            asum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            sum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce sqsum
    else if (param->type == 3)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += data[offset] * data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sqsum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sqsum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sqsum_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            sum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            sqsum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            sqsum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            sum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            sqsum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            sum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            sqsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            sqsum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce max
    else if (param->type == 4)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = FLOAT_MIN;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            if (data[offset] > s_tmp)
                                s_tmp = data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            max_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            max_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            max_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            max_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            max_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            max_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            max_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            max_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            max_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            max_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            max_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            max_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            max_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            max_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            max_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            max_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            max_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            max_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            max_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            max_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            max_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            max_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce min
    else if (param->type == 5)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = FLOAT_MAX;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            if (s_tmp > data[offset])
                                s_tmp = data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            min_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            min_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            min_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            min_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            min_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            min_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            min_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            min_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            min_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            min_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            min_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            min_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            min_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            min_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            min_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            min_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            min_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            min_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            min_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            min_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            min_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            min_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce prod
    else if (param->type == 6)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 1.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp *= data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            prod_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            prod_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            prod_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            prod_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            prod_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            prod_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            prod_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            prod_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            prod_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            prod_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            prod_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            prod_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            prod_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            prod_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            prod_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            prod_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            prod_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            prod_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            prod_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            prod_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            prod_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            prod_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce l1
    else if (param->type == 7)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += fabs(data[offset]);
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            asum_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            sum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            sum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            asum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            sum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            asum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            asum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce l2
    else if (param->type == 8)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += data[offset] * data[offset];
                        }
                    }
                }
            }
            tmp[0] = s_tmp;
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            l2_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            l2_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            l2_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            sum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            l2_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            l2_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            sum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            l2_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            sum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            l2_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            l2_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce log sum
    else if (param->type == 9)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += data[offset];
                        }
                    }
                }
            }
            tmp[0] = log(s_tmp);
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsum_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            logsum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            logsum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            logsum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            logsum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            logsum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            sum_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            logsum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            logsum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            logsum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            sum_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            logsum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            sum_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            logsum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    else if (param->type == 10)
    {
        if ((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
            (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for (int n = 0; n < dim0; n++)
            {
                for (int h = 0; h < dim1; h++)
                {
                    for (int w = 0; w < dim2; w++)
                    {
                        for (int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            s_tmp += exp(data[offset]);
                        }
                    }
                }
            }
            tmp[0] = log(s_tmp);
        }
        else if (param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsumexp_4d_ax1(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsumexp_4d_ax2(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            logsumexp_4d_ax3(dim0, dim1, dim2, dim3, data, tmp);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            logsum_3d_ax0(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            logsum_3d_ax1(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_03);
            logsum_3d_ax2(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            sumexp_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            logsum_3d_ax1(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            sumexp_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_13);
            logsum_3d_ax2(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if (param_dim2 == -2 && param_dim3 == -2 &&
                 ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            sumexp_4d_ax2(dim0, dim1, dim2, dim3, data, tmp_23);
            logsum_3d_ax2(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_0, tmp_01);
            logsum_2d_ax0(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_01);
            sum_3d_ax0(dim1, dim2, dim3, tmp_1, tmp_01);
            logsum_2d_ax1(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                      (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            sumexp_4d_ax0(dim0, dim1, dim2, dim3, data, tmp_02);
            sum_3d_ax1(dim1, dim2, dim3, tmp_1, tmp_02);
            logsum_2d_ax1(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if (param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                      (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                      (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                      (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                      (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                      (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            sumexp_4d_ax1(dim0, dim1, dim2, dim3, data, tmp_12);
            sum_3d_ax1(dim0, dim2, dim3, tmp_1, tmp_12);
            logsum_2d_ax1(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // pase to out_data
    for (int i = 0; i < out_size; i++)
    {
        out_data[i] = tmp[i];
    }
    sys_free(tmp);
    return 0;
}
// mean
void mean_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        float s_tmp = 0.f;
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            s_tmp += data[offset];
        }
        tmp[j] = s_tmp / dim0;
    }
}
void mean_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            float s_tmp = 0.f;
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                s_tmp += data[offset];
            }
            tmp[n * dim2 * dim3 + cw] = s_tmp / dim1;
        }
    }
}
void mean_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                float s_tmp = 0.f;
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    s_tmp += data[offset];
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = s_tmp / dim2;
            }
        }
    }
}
void mean_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                float s_tmp = 0.f;
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    s_tmp += data[offset];
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = s_tmp / dim3;
            }
        }
    }
}
void mean_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        float s_tmp = 0.f;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            s_tmp += tmp_01[index];
        }
        tmp[wc] = s_tmp / dim1;
    }
}
void mean_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            float s_tmp = 0.f;
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                s_tmp += tmp_02[index];
            }
            tmp[h * dim3 + c] = s_tmp / dim2;
        }
    }
}
void mean_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            float s_tmp = 0.f;
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                s_tmp += tmp_03[index];
            }
            tmp[h * dim2 + w] += s_tmp / dim3;
        }
    }
}
void mean_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        float s_tmp = 0.f;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            s_tmp += tmp_0[index];
        }
        tmp[w] += s_tmp / dim1;
    }
}
void mean_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        float s_tmp = 0.f;
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            s_tmp += tmp_1[index];
        }
        tmp[h] += s_tmp / dim2;
    }
}

// sum
void sum_5d_ax1(int* dims, int dim_num, float* data, float* tmp)
{
    int dim0 = dims[0];
    int dim1 = dims[1];
    int dim2 = dims[2];
    int dim3 = dims[3];
    int dim4 = dims[4];
    int chw = dim2*dim3*dim4;
    for(int j = 0; j < dim0; j++){
        for(int n = 0; n < dim1; n++){
            for(int size = 0; size < chw; size++){
                tmp[size] += data[n*chw + size];
            }
        }
    }
}

void sum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += data[offset];
        }
    }
}
void sum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += data[offset];
            }
        }
    }
}
void sum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += data[offset];
                }
            }
        }
    }
}
void sum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += data[offset];
                }
            }
        }
    }
}
void sum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += tmp_01[index];
        }
    }
}
void sum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += tmp_02[index];
            }
        }
    }
}
void sum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += tmp_03[index];
            }
        }
    }
}
void sum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += tmp_0[index];
        }
    }
}
void sum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += tmp_1[index];
        }
    }
}

// asum
void asum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += fabs(data[offset]);
        }
    }
}
void asum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += fabs(data[offset]);
            }
        }
    }
}
void asum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += fabs(data[offset]);
                }
            }
        }
    }
}
void asum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += fabs(data[offset]);
                }
            }
        }
    }
}
void asum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += fabs(tmp_01[index]);
        }
    }
}
void asum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += fabs(tmp_02[index]);
            }
        }
    }
}
void asum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += fabs(tmp_03[index]);
            }
        }
    }
}
void asum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += fabs(tmp_0[index]);
        }
    }
}
void asum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += fabs(tmp_1[index]);
        }
    }
}

// sqsum
void sqsum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += data[offset] * data[offset];
        }
    }
}
void sqsum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += data[offset] * data[offset];
            }
        }
    }
}
void sqsum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += data[offset] * data[offset];
                }
            }
        }
    }
}
void sqsum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += data[offset] * data[offset];
                }
            }
        }
    }
}
void sqsum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += tmp_01[index] * tmp_01[index];
        }
    }
}
void sqsum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += tmp_02[index] * tmp_02[index];
            }
        }
    }
}
void sqsum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += tmp_03[index] * tmp_03[index];
            }
        }
    }
}
void sqsum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += tmp_0[index] * tmp_0[index];
        }
    }
}
void sqsum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += tmp_1[index] * tmp_1[index];
        }
    }
}

// max
void max_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        float temp = FLOAT_MIN;
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            if (data[offset] > temp)
                temp = data[offset];
        }
        tmp[j] = temp;
    }
}
void max_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            float temp = FLOAT_MIN;
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                if (data[offset] > temp)
                    temp = data[offset];
            }
            tmp[n * dim2 * dim3 + cw] = temp;
        }
    }
}
void max_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                float temp = FLOAT_MIN;
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    if (data[offset] > temp)
                        temp = data[offset];
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = temp;
            }
        }
    }
}
void max_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                float temp = FLOAT_MIN;
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    if (data[offset] > temp)
                        temp = data[offset];
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = temp;
            }
        }
    }
}
void max_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        float temp = FLOAT_MIN;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            if (tmp_01[index] > temp)
                temp = tmp_01[index];
        }
        tmp[wc] = temp;
    }
}
void max_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            float temp = FLOAT_MIN;
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                if (tmp_02[index] > temp)
                    temp = tmp_02[index];
            }
            tmp[h * dim3 + c] = temp;
        }
    }
}
void max_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            float temp = FLOAT_MIN;
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                if (tmp_03[index] > temp)
                    temp = tmp_03[index];
            }
            tmp[h * dim2 + w] = temp;
        }
    }
}
void max_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        float temp = FLOAT_MIN;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            if (tmp_0[index] > temp)
                temp = tmp_0[index];
        }
        tmp[w] = temp;
    }
}
void max_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        float temp = FLOAT_MIN;
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            if (tmp_1[index] > temp)
                temp = tmp_1[index];
        }
        tmp[h] = temp;
    }
}

// min
void min_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        float temp = FLOAT_MAX;
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            if (data[offset] < temp)
                temp = data[offset];
        }
        tmp[j] = temp;
    }
}
void min_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            float temp = FLOAT_MAX;
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                if (data[offset] < temp)
                    temp = data[offset];
            }
            tmp[n * dim2 * dim3 + cw] = temp;
        }
    }
}
void min_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                float temp = FLOAT_MAX;
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    if (data[offset] < temp)
                        temp = data[offset];
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = temp;
            }
        }
    }
}
void min_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                float temp = FLOAT_MAX;
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    if (data[offset] < temp)
                        temp = data[offset];
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = temp;
            }
        }
    }
}
void min_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        float temp = FLOAT_MAX;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            if (tmp_01[index] < temp)
                temp = tmp_01[index];
        }
        tmp[wc] = temp;
    }
}
void min_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            float temp = FLOAT_MAX;
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                if (tmp_02[index] < temp)
                    temp = tmp_02[index];
            }
            tmp[h * dim3 + c] = temp;
        }
    }
}
void min_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            float temp = FLOAT_MAX;
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                if (tmp_03[index] < temp)
                    temp = tmp_03[index];
            }
            tmp[h * dim2 + w] = temp;
        }
    }
}
void min_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        float temp = FLOAT_MAX;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            if (tmp_0[index] < temp)
                temp = tmp_0[index];
        }
        tmp[w] = temp;
    }
}
void min_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        float temp = FLOAT_MIN;
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            if (tmp_1[index] < temp)
                temp = tmp_1[index];
        }
        tmp[h] = temp;
    }
}

// prod
void prod_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        tmp[j] = 1.f;
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] *= data[offset];
        }
    }
}
void prod_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            tmp[n * dim2 * dim3 + cw] = 1.f;
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] *= data[offset];
            }
        }
    }
}
void prod_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                tmp[n * dim1 * dim3 + h * dim3 + c] = 1.f;
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] *= data[offset];
                }
            }
        }
    }
}
void prod_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                tmp[n * dim1 * dim2 + h * dim2 + w] = 1.f;
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] *= data[offset];
                }
            }
        }
    }
}
void prod_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        tmp[wc] = 1.f;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] *= tmp_01[index];
        }
    }
}
void prod_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            tmp[h * dim3 + c] = 1.f;
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] *= tmp_02[index];
            }
        }
    }
}
void prod_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            tmp[h * dim2 + w] = 1.f;
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] *= tmp_03[index];
            }
        }
    }
}
void prod_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        tmp[w] = 1.f;
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] *= tmp_0[index];
        }
    }
}
void prod_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        tmp[h] = 1.f;
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] *= tmp_1[index];
        }
    }
}

// l2
void l2_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += sqrt((double )data[offset] * data[offset]);
        }
    }
}
void l2_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += sqrt((double )data[offset] * data[offset]);
            }
        }
    }
}
void l2_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += sqrt((double )data[offset] * data[offset]);
                }
            }
        }
    }
}
void l2_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += sqrt((double )data[offset] * data[offset]);
                }
            }
        }
    }
}
void l2_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += sqrt((double )tmp_01[index] * tmp_01[index]);
        }
    }
}
void l2_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += sqrt((double )tmp_02[index] * tmp_02[index]);
            }
        }
    }
}
void l2_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += sqrt((double )tmp_03[index] * tmp_03[index]);
            }
        }
    }
}
void l2_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += sqrt((double )tmp_0[index] * tmp_0[index]);
        }
    }
}
void l2_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += sqrt((double )tmp_1[index] * tmp_1[index]);
        }
    }
}

// logsum
void logsum_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += data[offset];
        }
        tmp[j] = log(tmp[j]);
    }
}
void logsum_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += data[offset];
            }
            tmp[n * dim2 * dim3 + cw] = log(tmp[n * dim2 * dim3 + cw]);
        }
    }
}
void logsum_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += data[offset];
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = log(tmp[n * dim1 * dim3 + h * dim3 + c]);
            }
        }
    }
}
void logsum_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += data[offset];
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = log(tmp[n * dim1 * dim2 + h * dim2 + w]);
            }
        }
    }
}
void logsum_3d_ax0(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for (int wc = 0; wc < dim2 * dim3; wc++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += tmp_01[index];
        }
        tmp[wc] = log(tmp[wc]);
    }
}
void logsum_3d_ax1(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int c = 0; c < dim3; c++)
        {
            for (int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += tmp_02[index];
            }
            tmp[h * dim3 + c] = log(tmp[h * dim3 + c]);
        }
    }
}
void logsum_3d_ax2(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            for (int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += tmp_03[index];
            }
            tmp[h * dim2 + w] = log(tmp[h * dim2 + w]);
        }
    }
}
void logsum_2d_ax0(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for (int w = 0; w < dim2; w++)
    {
        for (int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += tmp_0[index];
        }
        tmp[w] = log(tmp[w]);
    }
}
void logsum_2d_ax1(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for (int h = 0; h < dim1; h++)
    {
        for (int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += tmp_1[index];
        }
        tmp[h] = log(tmp[h]);
    }
}

// logsumexp
void logsumexp_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += exp(data[offset]);
        }
        tmp[j] = log(tmp[j]);
    }
}
void logsumexp_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += exp(data[offset]);
            }
            tmp[n * dim2 * dim3 + cw] = log(tmp[n * dim2 * dim3 + cw]);
        }
    }
}
void logsumexp_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += exp(data[offset]);
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = log(tmp[n * dim1 * dim3 + h * dim3 + c]);
            }
        }
    }
}
void logsumexp_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += exp(data[offset]);
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = log(tmp[n * dim1 * dim2 + h * dim2 + w]);
            }
        }
    }
}
void sumexp_4d_ax0(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for (int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            tmp[j] += exp(data[offset]);
        }
        tmp[j] = tmp[j];
    }
}
void sumexp_4d_ax1(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int cw = 0; cw < dim2 * dim3; cw++)
        {
            for (int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                tmp[n * dim2 * dim3 + cw] += exp(data[offset]);
            }
            tmp[n * dim2 * dim3 + cw] = tmp[n * dim2 * dim3 + cw];
        }
    }
}
void sumexp_4d_ax2(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim3 + h * dim3 + c] += exp(data[offset]);
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = tmp[n * dim1 * dim3 + h * dim3 + c];
            }
        }
    }
}
void sumexp_4d_ax3(int dim0, int dim1, int dim2, int dim3, float* data, float* tmp)
{
    for (int n = 0; n < dim0; n++)
    {
        for (int h = 0; h < dim1; h++)
        {
            for (int w = 0; w < dim2; w++)
            {
                for (int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    tmp[n * dim1 * dim2 + h * dim2 + w] += exp(data[offset]);
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = tmp[n * dim1 * dim2 + h * dim2 + w];
            }
        }
    }
}
