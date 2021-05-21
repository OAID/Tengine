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
 * Author: lswang@openailab.com
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

#include "trt_define.h"

EXPORT_BEGIN
#include "operator/op.h"
EXPORT_FINISH

#include <NvInfer.h>


#if NV_TENSORRT_MAJOR < 5
#error "Tengine: The minimum supported version of TensorRT is 5.\n"
#endif

const int trt_supported_ops[] = {
        OP_ABSVAL,
        OP_ADD_N,
#if NV_TENSORRT_MAJOR >= 6
        OP_ARGMAX,
        OP_ARGMIN,
#endif
        OP_BATCHNORM,
        //OP_BATCHTOSPACEND,            // Not supported, last checked version 7.1.3
        OP_BIAS,
#if NV_TENSORRT_MAJOR >= 6
        OP_BROADMUL,
        OP_CAST,
        OP_CEIL,
        OP_CLIP,
#endif
#if NV_TENSORRT_MAJOR >= 7
        OP_COMPARISON,
#endif
        OP_CONCAT,
        OP_CONST,
        OP_CONV,
        OP_CROP,
        OP_DECONV,
        OP_DEPTHTOSPACE,
        //OP_DETECTION_OUTPUT,          // Not supported, last checked version 7.1.3
        //OP_DETECTION_POSTPROCESS,     // Not supported, last checked version 7.1.3
        OP_DROPOUT,
        OP_ELTWISE,
        OP_ELU,
        //OP_EMBEDDING,                 // Not supported, last checked version 7.1.3
#if NV_TENSORRT_MAJOR >= 6
        OP_EXPANDDIMS,
#endif
        OP_FC,
        OP_FLATTEN,
        OP_GATHER,
        OP_GEMM,
#if NV_TENSORRT_MAJOR >= 7
        OP_GRU,
#endif
        OP_HARDSIGMOID,
        OP_HARDSWISH,                   // Not supported, last checked version 7.1.3
        OP_INPUT,
        OP_INSTANCENORM,
        OP_INTERP,                      // should be as UpSample
        OP_LOGICAL,
#if NV_TENSORRT_MAJOR >= 7
        OP_LOGISTIC,
#endif
        OP_LRN,
#if NV_TENSORRT_MAJOR >= 7
        OP_LSTM,
#endif
        OP_MATMUL,
        OP_MAXIMUM,
        OP_MEAN,
        OP_MINIMUM,
        //OP_MVN,                       // Not supported, last checked version 7.1.3
        OP_NOOP,
        //OP_NORMALIZE,                 // Not supported, last checked version 7.1.3
        OP_PAD,
        OP_PERMUTE,
        OP_POOL,
        OP_PRELU,
        //OP_PRIORBOX,                  // Not supported, last checked version 7.1.3
        //OP_PSROIPOOLING,              // Not supported, last checked version 7.1.3
        OP_REDUCEL2,
        OP_REDUCTION,
        //OP_REGION,                    // Not supported, last checked version 7.1.3
        OP_RELU,
        OP_RELU6,
        //OP_REORG,                     // Not supported, last checked version 7.1.3
        OP_RESHAPE,
#if NV_TENSORRT_MAJOR >= 6
        OP_RESIZE,
#endif
        //OP_REVERSE,                   // Not supported, last checked version 7.1.3
#if NV_TENSORRT_MAJOR >= 7
        OP_RNN,
#endif
        //OP_ROIALIGN,                  // Not supported, last checked version 7.1.3
        //OP_ROIPOOLING,                // Not supported, last checked version 7.1.3
        //OP_ROUND,
        //OP_RPN,
        OP_SCALE,
        OP_SELU,
        //OP_SHUFFLECHANNEL,            // Not supported, last checked version 7.1.3
        OP_SIGMOID,
#if NV_TENSORRT_MAJOR >= 6
        OP_SLICE,
#endif
        OP_SOFTMAX,
        //OP_SPACETOBATCHND,            // Not supported, last checked version 7.1.3
        OP_SPACETODEPTH,
        //OP_SPARSETODENSE,             // Not supported, last checked version 7.1.3
        OP_SPLIT,
        //OP_SQUAREDDIFFERENCE,         // Not supported, last checked version 7.1.3
        OP_SQUEEZE,
        //OP_STRIDED_SLICE,             // Not supported, last checked version 7.1.3
        //OP_SWAP_AXIS,
        OP_TANH,
        //OP_THRESHOLD,                 // Not supported, last checked version 7.1.3
        //OP_THRESHOLD,                 // Not supported, last checked version 7.1.3
        OP_TOPKV2,
        OP_TRANSPOSE,
        OP_UNARY,
        OP_UNSQUEEZE,
        OP_UPSAMPLE,
        //OP_ZEROSLIKE,                 // Not supported, last checked version 7.1.3
        OP_MISH,
        OP_LOGSOFTMAX,
#if NV_TENSORRT_MAJOR >= 6
        OP_RELU1,
#endif
        //OP_L2NORMALIZATION,         // Not supported, last checked version 7.1.3
        //OP_L2POOL,                  // Not supported, last checked version 7.1.3
#if NV_TENSORRT_MAJOR >= 7
        OP_TILE,
#endif
        OP_SHAPE,
        OP_SCATTER,
#if NV_TENSORRT_MAJOR >= 7
        OP_WHERE,
#endif
};
