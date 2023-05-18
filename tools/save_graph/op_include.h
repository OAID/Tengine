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
 * Copyright (c) 2020, Open AI Lab
 * Author: bhu@openailab.com
 */

#ifndef __OP_INCLUDE_H__
#define __OP_INCLUDE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "operator/op.h"
#include "argmax_param.h"
#include "deconv_param.h"
#include "gather_param.h"
#include "lstm_param.h"
#include "region_param.h"
#include "selu_param.h"
#include "swap_axis_param.h"
#include "argmin_param.h"
#include "depthtospace_param.h"
#include "gemm_param.h"
#include "mvn_param.h"
#include "relu_param.h"
#include "shuffle_channel_param.h"
#include "threshold_param.h"
#include "batchnorm_param.h"
#include "detection_output_param.h"
#include "generic_param.h"
#include "normalize_param.h"
#include "reorg_param.h"
#include "slice_param.h"
#include "topkv2_param.h"
#include "batchtospacend_param.h"
#include "detection_postprocess_param.h"
#include "gru_param.h"
#include "pad_param.h"
#include "reshape_param.h"
#include "softmax_param.h"
#include "transpose_param.h"
#include "cast_param.h"
#include "eltwise_param.h"
#include "hardsigmoid_param.h"
#include "permute_param.h"
#include "resize_param.h"
#include "spacetobatchnd_param.h"
#include "unary_param.h"
#include "clip_param.h"
#include "elu_param.h"
#include "hardswish_param.h"
#include "pooling_param.h"
#include "rnn_param.h"
#include "spacetodepth_param.h"
#include "unsqueeze_param.h"
#include "comparison_param.h"
#include "embedding_param.h"
#include "instancenorm_param.h"
#include "priorbox_param.h"
#include "roialign_param.h"
#include "sparsetodense_param.h"
#include "upsample_param.h"
#include "concat_param.h"
#include "expanddims_param.h"
#include "interp_param.h"
#include "psroipooling_param.h"
#include "roipooling_param.h"
#include "split_param.h"
#include "convolution_param.h"
#include "fc_param.h"
#include "logical_param.h"
#include "reducel2_param.h"
#include "rpn_param.h"
#include "squeeze_param.h"
#include "crop_param.h"
#include "flatten_param.h"
#include "lrn_param.h"
#include "reduction_param.h"
#include "scale_param.h"
#include "strided_slice_param.h"
#include "logsoftmax_param.h"
#include "scatter_param.h"
#include "hardsigmoid_param.h"
#include "tile_param.h"
#include "expand_param.h"
#include "spatialtransformer_param.h"
#include "layernorm_param.h"

#ifdef __cplusplus
}
#endif

#endif
