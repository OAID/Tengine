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

int ref_layernormlstm_fp16(__fp16* total_input_data, __fp16* total_output_data, lnlstm_param* param)
{
    /* for arm32 && x86 */
    for(int i = 0; i < param->sequence_size; i++)
    {
        float* input_data = ( float* )malloc(param->batch_size * param->input_size * sizeof(float));
        float* output_data = ( float* )malloc(param->batch_size * param->output_size * sizeof(float));
        memset(output_data, 0, param->batch_size * param->output_size * sizeof(float));
        int per_batch_num = param->input_size * param->sequence_size;
        for(int j = 0; j < param->batch_size; j++)
        {
            for(int k = 0; k < param->input_size; k++)
            {
                input_data[j * param->input_size + k] =
                    fp16_to_fp32(total_input_data[per_batch_num * j + i * param->input_size + k]);
            }
        }

        DoLayerNormLSTM(
            input_data, output_data, param->i2i_weights_data, param->i2c_weights_data, param->i2f_weights_data,
            param->i2o_weights_data, param->r2i_weights_data, param->r2c_weights_data, param->r2f_weights_data,
            param->r2o_weights_data, param->c2i_weights_data, param->c2f_weights_data, param->c2o_weights_data,
            param->igate_bias_data, param->cgate_bias_data, param->fgate_bias_data, param->ogate_bias_data,
            param->projection_bias_data, param->projection_weights_data, param->icellstate_data,
            param->iactivationstate_data, param->ilayer_norm_coefficients_data, param->flayer_norm_coefficients_data,
            param->clayer_norm_coefficients_data, param->olayer_norm_coefficients_data, param->batch_size,
            param->cell_size, param->input_size, param->output_size, param->output_true_size, param->fused_activation,
            param->cell_clip, param->proj_clip, param->input_gate_scratch, param->forget_gate_scratch,
            param->cell_scratch, param->output_gate_scratch);
        int per_batch_num_o = param->output_size * param->sequence_size;
        for(int m = 0; m < param->batch_size; m++)
        {
            for(int n = 0; m < param->output_size; m++)
            {
                total_output_data[per_batch_num_o * m + i * param->output_size + n] =
                    output_data[m * param->output_size + n];
            }
        }
        free(input_data);
        free(output_data);
    }
    return 0;
}
