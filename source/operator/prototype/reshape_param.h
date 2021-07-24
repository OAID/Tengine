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
 * Author: haitao@openailab.com
 */

#ifndef __RESHAPE_PARAM_H__
#define __RESHAPE_PARAM_H__

typedef struct reshape_param
{
    /*
    int dim_0;
    int dim_1;
    int dim_2;
    int dim_3;
    int dim_size;
    int axis;
    */
    int* re_shape;
    int reverse;
    int is_mxnet;
    int is_onnx;
    int dim_size;
} reshape_param_t;

#endif
