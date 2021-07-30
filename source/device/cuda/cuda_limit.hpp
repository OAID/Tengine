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
 * Copyright (c) 2021, Open AI Lab
 * Author: hhchen@openailab.com
 */

// nVIDIA Jetson version matrix
// JetPack    CUDA        GCC    cuDNN      TensorRT    OpenCV    MultiMedia API  DeepStream
//   4.1.1    10.0               7.3.1      5.0.3
//   4.2      10.166             7.3.1.28   5.0.6.3                   2.1
//   4.2.1    10.0.326    7.3    7.5.0.56   5.1.6.1                                  4.0
//   4.3      10.0.326           7.6.3      6.0.1.10    4.1.1                        4.0.2
//   4.4      10.2               8.0.0      7.1.3

// NV_TENSORRT_MAJOR NV_TENSORRT_MINOR NV_TENSORRT_PATCH NV_TENSORRT_BUILD

#pragma once

extern "C" {
#include "operator/op.h"
}

const int cuda_supported_ops[] = {
    OP_CLIP,
    OP_CONCAT,
    OP_CONST,
    OP_CONV,
    OP_DROPOUT,
    OP_ELTWISE,
    OP_FC,
    OP_FLATTEN,
    OP_INPUT,
    OP_PERMUTE,
    OP_POOL,
    OP_RELU,
    OP_RESHAPE,
    OP_SLICE,
    OP_SOFTMAX};
