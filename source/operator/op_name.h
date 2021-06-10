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

#pragma once

#define OP_GENERIC_NAME                             "Generic"
#define OP_ABSVAL_NAME                              "Absval"
#define OP_ADD_N_NAME                               "Add_n"
#define OP_ARGMAX_NAME                              "ArgMax"
#define OP_ARGMIN_NAME                              "ArgMin"
#define OP_BATCHNORM_NAME                           "BatchNormalize"
#define OP_BATCHTOSPACEND_NAME                      "Batchtospacend"
#define OP_BIAS_NAME                                "Bias"
#define OP_BROADMUL_NAME                            "BroadMul"
#define OP_CAST_NAME                                "Cast"
#define OP_CEIL_NAME                                "Ceil"
#define OP_CLIP_NAME                                "Clip"
#define OP_COMPARISON_NAME                          "Comparison"
#define OP_CONCAT_NAME                              "Concat"
#define OP_CONV_NAME                                "Convolution"
#define OP_CONST_NAME                               "Const"
#define OP_CROP_NAME                                "Crop"
#define OP_DECONV_NAME                              "Deconvolution"
#define OP_DEPTHTOSPACE_NAME                        "Depthtospace"
#define OP_DETECTION_OUTPUT_NAME                    "DetectionOutput"
#define OP_DETECTION_POSTPROCESS_NAME               "DetectionPostProcess"
#define OP_DROPOUT_NAME                             "Dropout"
#define OP_ELTWISE_NAME                             "Eltwise"
#define OP_ELU_NAME                                 "Elu"
#define OP_EMBEDDING_NAME                           "Embedding"
#define OP_EXPANDDIMS_NAME                          "Expanddims"
#define OP_FC_NAME                                  "FullyConnected"
#define OP_FLATTEN_NAME                             "Flatten"
#define OP_GATHER_NAME                              "Gather"
#define OP_GEMM_NAME                                "Gemm"
#define OP_GRU_NAME                                 "Gru"
#define OP_HARDSIGMOID_NAME                         "HardSigmoid"
#define OP_HARDSWISH_NAME                           "Hardswish"
#define OP_INPUT_NAME                               "InputOp"
#define OP_INSTANCENORM_NAME                        "InstanceNorm"
#define OP_INTERP_NAME                              "Interp"
#define OP_LOGICAL_NAME                             "Logical"
#define OP_LOGISTIC_NAME                            "Logistic"
#define OP_LRN_NAME                                 "Lrn"
#define OP_LSTM_NAME                                "Lstm"
#define OP_MATMUL_NAME                              "Matmul"
#define OP_MAXIMUM_NAME                             "Maximum"
#define OP_MEAN_NAME                                "Mean"
#define OP_MINIMUM_NAME                             "Minimum"
#define OP_MVN_NAME                                 "Mvn"
#define OP_NOOP_NAME                                "Noop"
#define OP_NORMALIZE_NAME                           "Normalize"
#define OP_PAD_NAME                                 "Pad"
#define OP_PERMUTE_NAME                             "Permute"
#define OP_POOL_NAME                                "Pooling"
#define OP_PRELU_NAME                               "PReLU"
#define OP_PRIORBOX_NAME                            "PriorBox"
#define OP_PSROIPOOLING_NAME                        "Psroipooling"
#define OP_REDUCEL2_NAME                            "ReduceL2"
#define OP_REDUCTION_NAME                           "Reduction"
#define OP_REGION_NAME                              "Region"
#define OP_RELU_NAME                                "ReLU"
#define OP_RELU6_NAME                               "ReLU6"
#define OP_REORG_NAME                               "Reorg"
#define OP_RESHAPE_NAME                             "Reshape"
#define OP_RESIZE_NAME                              "Resize"
#define OP_REVERSE_NAME                             "Reverse"
#define OP_RNN_NAME                                 "RNN"
#define OP_ROIALIGN_NAME                            "Roialign"
#define OP_ROIPOOLING_NAME                          "RoiPooling"
#define OP_ROUND_NAME                               "Round"
#define OP_RPN_NAME                                 "Rpn"
#define OP_SCALE_NAME                               "Scale"
#define OP_SELU_NAME                                "Selu"
#define OP_SHUFFLECHANNEL_NAME                      "ShuffleChannel"
#define OP_SIGMOID_NAME                             "Sigmoid"
#define OP_SLICE_NAME                               "Slice"
#define OP_SOFTMAX_NAME                             "Softmax"
#define OP_SPACETOBATCHND_NAME                      "Spacetobatchnd"
#define OP_SPACETODEPTH_NAME                        "Spacetodepth"
#define OP_SPARSETODENSE_NAME                       "SparseToDense"
#define OP_SPLIT_NAME                               "Split"
#define OP_SQUAREDDIFFERENCE_NAME                   "SquaredDifference"
#define OP_SQUEEZE_NAME                             "Squeeze"
#define OP_STRIDEDSLICE_NAME                        "StridedSlice"
#define OP_SWAP_AXIS_NAME                           "SwapAxis"
#define OP_TANH_NAME                                "Tanh"
#define OP_THRESHOLD_NAME                           "Threshold"
#define OP_TOPKV2_NAME                              "Topkv2"
#define OP_TRANSPOSE_NAME                           "Transpose"
#define OP_UNARY_NAME                               "Unary"
#define OP_UNSQUEEZE_NAME                           "Unsqueeze"
#define OP_UPSAMPLE_NAME                            "Upsample"
#define OP_ZEROSLIKE_NAME                           "ZerosLike"
#define OP_MISH_NAME                                "Mish"
#define OP_LOGSOFTMAX_NAME                          "LogSoftmax"
#define OP_RELU1_NAME                               "ReLU1"
#define OP_L2NORMALIZATION_NAME                     "L2Normalization"
#define OP_L2POOL_NAME                              "L2Pool"
#define OP_TILE_NAME                                "Tile"
#define OP_SHAPE_NAME                               "Shape"
#define OP_SCATTER_NAME                             "Scatter"
#define OP_WHERE_NAME                               "Where"
#define OP_SOFTPLUS_NAME                            "Softplus"
#define OP_RECIPROCAL_NAME                          "Reciprocal"
#define OP_SPATIALTRANSFORMER_NAME                  "SpatialTransformer"
#define OP_EXPAND_NAME                              "Expand"
