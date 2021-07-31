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
 * Author: zpluo@openailab.com
 */

#ifndef __LSTM_PARAM_H__
#define __LSTM_PARAM_H__

#define LSTM_ACT_SIGMOID 1
#define LSTM_ACT_TANH    2
typedef struct lstm_param
{
    float forget_bias;
    float clip;
    int output_len;
    int sequence_len;
    int input_size;
    int hidden_size;
    int cell_size;
    int has_peephole;
    int has_projection;
    int has_clip;
    int has_bias;
    int has_init_state;
    int forget_act;
    int input_act;
    int output_act;
    int cellin_act;
    int cellout_act;
    int mxnet_flag;
} lstm_param_t;

#endif
