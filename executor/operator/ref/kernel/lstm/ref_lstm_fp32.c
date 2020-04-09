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
 * Copyright (c) 2019, Open AI Lab
 * Author: zpluo@openailab.com
 */

int ref_lstm_fp32(float* input,float* output,lstm_param* param)
{

    float* init_h = ( float* )malloc(param->batch_size * param->hidden_size * sizeof(float));


    float* init_c = ( float* )malloc(param->batch_size * param->cell_size * sizeof(float));


    if(param->init_h_data)
    {
        for(int i = 0; i < param->batch_size; i++)
        {
            memcpy(init_h + i * param->hidden_size, param->init_h_data, param->hidden_size * sizeof(float));
            memcpy(init_c + i * param->cell_size, param->init_c_data, param->cell_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(param->batch_size * param->hidden_size * sizeof(float)));
        memset(init_c, 0x0, sizeof(param->batch_size * param->cell_size * sizeof(float)));
    }
    for(int i = 0; i < param->seq_lens; i++)
    {
        const float* seq_input = input + i * param->batch_size * param->input_size;

        if(!do_LSTM_step(seq_input, init_h, init_c, param->kernel, param->bias, param->h2h_kernel, param->h2h_bias, param->w_f_data, param->w_i_data,
                            param->w_o_data, param->projection, param->forget_bias, param->batch_size, param->input_size, param->hidden_size, param->cell_size,
                            param->mxnet_flag))
            return false;

        if(i + param->output_len >= param->seq_lens)
        {
            memcpy(output, init_h, param->hidden_size * param->batch_size * sizeof(float));
            output += param->batch_size * param->hidden_size;
        }
    }
    free(init_h);
    free(init_c);
    return 0;
}
