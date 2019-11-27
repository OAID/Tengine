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
 * Author: jingyou@openailab.com
 */

static int ref_lrn_int8(const int8_t* in_data, int8_t* out_data, ref_lrn_param* param)
{
    int n = param->dims[0];
    int c = param->dims[1];
    int h = param->dims[2];
    int w = param->dims[3];

    float alpha = param->alpha;
    float beta = param->beta;
    float bias = param->bias;
    int local_size = param->local_size;

    int channel_size = h * w;
    int img_size = c * channel_size;
    int input_size = n * img_size;

    float* square = ( float* )(malloc(img_size * sizeof(float)));
    float* accum_square = ( float* )(malloc(channel_size * sizeof(float)));
    float* input_f = ( float* )(malloc(input_size * sizeof(float)));
    float* output_f = ( float* )(malloc(input_size * sizeof(float)));

    for(int i = 0; i < input_size; i++)
        input_f[i] = in_data[i] * param->scale[0];

    for(int i = 0; i < n; i++)
    {
        const float* img_base = input_f + i * img_size;

        /* get square value */
        for(int j = 0; j < img_size; j++)
            square[j] = img_base[j] * img_base[j] + bias;

        if(param->norm_region == 0) /* LRN_ACROSS_CHANNELS */
        {
            float alpha_over_size = alpha / local_size;

            for(int j = 0; j < c; j++)
            {
                int c_start = j - local_size / 2;
                int c_end = j + local_size / 2;

                memset(accum_square, 0x0, channel_size * sizeof(float));

                for(int l = c_start; l <= c_end; l++)
                {
                    if(l < 0 || l >= c)
                        continue;

                    for(int n = 0; n < channel_size; n++)
                    {
                        accum_square[n] += square[l * channel_size + n];
                    }
                }

                /* get the output */
                for(int n = 0; n < channel_size; n++)
                {
                    int offset = i * img_size + j * channel_size + n;
                    output_f[offset] = input_f[offset] * pow(1.0f + alpha_over_size * accum_square[n], -beta);
                }
            }
        }
        else
        {
            free(square);
            free(accum_square);
            free(input_f);
            free(output_f);
            return -1;
        }
    }

    float max_val = 0.0f;
    for(int i = 0; i < input_size; i++)
    {
        if(max_val < fabs(output_f[i]))
            max_val = fabs(output_f[i]);
    }
    float out_scale = max_val / 127;
    for(int i = 0; i < input_size; i++)
    {
        out_data[i] = (int8_t)(round(output_f[i] / out_scale));
    }
    param->scale[1] = out_scale;

    free(square);
    free(accum_square);
    free(input_f);
    free(output_f);
    return 0;
}
