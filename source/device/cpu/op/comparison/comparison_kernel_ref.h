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
 * Author: ddzhao@openailab.com
 */

#ifndef __COMPARISON_KERNEL_REF_H__
#define __COMPARISON_KERNEL_REF_H__

typedef struct __comparison_param
{
    int type;
    int layout;
    int shape0[4];
    int shape1[4];
} _comparison_param, *p_comparison_param;

extern int ref_comparison_fp32(float* input0, float* input1, float* output, p_comparison_param param);

#endif
