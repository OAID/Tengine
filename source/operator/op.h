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
 * Revised: lswang@openailab.com
 */

#pragma once

#include <stdint.h>

#include "op_name.h"


struct node;


/*!
 * @enum  op_type
 * @brief Enumeration of supported operators
 */
enum
{
    OP_GENERIC = 0,
    OP_ABSVAL,
    OP_ADD_N,
    OP_ARGMAX,
    OP_ARGMIN,
    OP_BATCHNORM,
    OP_BATCHTOSPACEND,
    OP_BIAS,
    OP_BROADMUL,
    OP_CAST,
    OP_CEIL,
    OP_CLIP,
    OP_COMPARISON,
    OP_CONCAT,
    OP_CONST,    
    OP_CONV,
    OP_CROP,
    OP_DECONV,
    OP_DEPTHTOSPACE,
    OP_DETECTION_OUTPUT,
    OP_DETECTION_POSTPROCESS,
    OP_DROPOUT,
    OP_ELTWISE,
    OP_ELU,
    OP_EMBEDDING,
    OP_EXPANDDIMS,
    OP_FC,
    OP_FLATTEN,
    OP_GATHER,
    OP_GEMM,
    OP_GRU,
    OP_HARDSIGMOID,
    OP_HARDSWISH,
    OP_INPUT,
    OP_INSTANCENORM,
    OP_INTERP,
    OP_LOGICAL,
    OP_LOGISTIC,
    OP_LRN,
    OP_LSTM,
    OP_MATMUL,
    OP_MAXIMUM,
    OP_MEAN,
    OP_MINIMUM,
    OP_MVN,
    OP_NOOP,
    OP_NORMALIZE,
    OP_PAD,
    OP_PERMUTE,
    OP_POOL,
    OP_PRELU,
    OP_PRIORBOX,
    OP_PSROIPOOLING,
    OP_REDUCEL2,
    OP_REDUCTION,
    OP_REGION,
    OP_RELU,
    OP_RELU6,
    OP_REORG,
    OP_RESHAPE,
    OP_RESIZE,
    OP_REVERSE,
    OP_RNN,
    OP_ROIALIGN,
    OP_ROIPOOLING,
    OP_ROUND,
    OP_RPN,
    OP_SCALE,
    OP_SELU,
    OP_SHUFFLECHANNEL,
    OP_SIGMOID,
    OP_SLICE,
    OP_SOFTMAX,
    OP_SPACETOBATCHND,
    OP_SPACETODEPTH,
    OP_SPARSETODENSE,
    OP_SPLIT,
    OP_SQUAREDDIFFERENCE,
    OP_SQUEEZE,
    OP_STRIDED_SLICE,
    OP_SWAP_AXIS,
    OP_TANH,
    OP_THRESHOLD,
    OP_TOPKV2,
    OP_TRANSPOSE,
    OP_UNARY,
    OP_UNSQUEEZE,
    OP_UPSAMPLE,
    OP_ZEROSLIKE,
    OP_MISH,
    OP_LOGSOFTMAX,
    OP_RELU1,
    OP_L2NORMALIZATION,
    OP_L2POOL,
    OP_TILE,
    OP_SHAPE,
    OP_SCATTER,
    OP_WHERE,
    OP_SOFTPLUS,
    OP_RECIPROCAL,
    OP_SPATIALTRANSFORMER,
    OP_EXPAND,
    OP_BUILTIN_LAST
};


/*!
 * @struct ir_op_t
 * @brief  Abstract operator intermediate representation
 */
typedef struct op
{
    uint16_t type;                          //!< the type of a operator
    uint8_t  version;                       //!< the version of a operator
    uint8_t  same_shape;                    //!< the flag of whether the operator will keep shape
    uint16_t param_size;                    //!< size of parameter memory buffer
    void* param_mem;                        //!< parameter memory buffer
    int (*infer_shape)(struct node*);       //!< infer(or broadcast) the shape from input to output(s)
} ir_op_t;


/*!
 * @struct ir_op_method_t
 * @brief  Abstract method of operator intermediate representation
 */
typedef struct method
{

    int  type;                              //!< the type of a operator
    int  version;                           //!< the version of a operator
    int  (*init)(ir_op_t* op);
    void (*release)(ir_op_t* op);
} ir_method_t;


void init_op_struct(ir_op_t* op);


void init_method_struct(ir_method_t* method);
