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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __LSTM_PARAM_HPP__
#define __LSTM_PARAM_HPP__

#include <vector>

#include "parameter.hpp"

namespace TEngine {

#define LSTM_ACT_SIGMOID 1
#define LSTM_ACT_TANH 2

struct LSTMParam : public NamedParam
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

    DECLARE_PARSER_STRUCTURE(LSTMParam)
    {
        DECLARE_PARSER_ENTRY(forget_bias);
        DECLARE_PARSER_ENTRY(clip);
        DECLARE_PARSER_ENTRY(output_len);
        DECLARE_PARSER_ENTRY(sequence_len);
        DECLARE_PARSER_ENTRY(input_size);
        DECLARE_PARSER_ENTRY(hidden_size);
        DECLARE_PARSER_ENTRY(cell_size);
        DECLARE_PARSER_ENTRY(has_peephole);
        DECLARE_PARSER_ENTRY(has_projection);
        DECLARE_PARSER_ENTRY(has_clip);
        DECLARE_PARSER_ENTRY(has_bias);
        DECLARE_PARSER_ENTRY(has_init_state);
        DECLARE_PARSER_ENTRY(forget_act);
        DECLARE_PARSER_ENTRY(input_act);
        DECLARE_PARSER_ENTRY(cellin_act);
        DECLARE_PARSER_ENTRY(output_act);
        DECLARE_PARSER_ENTRY(cellout_act);
        DECLARE_PARSER_ENTRY(mxnet_flag);
    };
};

}    // namespace TEngine

#endif
