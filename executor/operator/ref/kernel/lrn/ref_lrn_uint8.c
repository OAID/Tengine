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

static int ref_lrn_uint8(const uint8_t* in_data, uint8_t* out_data, ref_lrn_param* param)
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

    float* square = ( float* )(malloc(img_size * sizeof(float)));
    float* accum_square = ( float* )(malloc(channel_size * sizeof(float)));

    for(int i = 0; i < n; i++)
    {
        const uint8_t* img_base = in_data + i * img_size;

        /* get square value */
        for(int j = 0; j < img_size; j++)
        {
            float img_data = (img_base[j] - param->zero[0]) * param->scale[0];
            square[j] = img_data * img_data + bias;
        }

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
                    float output_f = in_data[offset] * pow(1.0f + alpha_over_size * accum_square[n], -beta);
                    out_data[offset] = (uint8_t)(round(output_f / param->scale[1]) + param->zero[1]);
                }
            }
        }
        else
        {
            free(square);
            free(accum_square);
            return -1;
        }
    }

    free(square);
    free(accum_square);
    return 0;
}
