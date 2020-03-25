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

int ref_l2norm_fp32(float* data,float* out_data, int size,int channel_size, l2norm_param* param)
{
    for(int i = 0; i < size; i++)
    {
        float sq_l2_norm = 0;
        for(int j = 0; j < channel_size; j++)
        {
            const float val = data[j];
            sq_l2_norm += val * val;
            //std::cout<<sq_l2_norm<<std::endl;
        }
        const float l2_norm = sqrt(sq_l2_norm);
        for(int j = 0; j < channel_size; j++)
        {
            *out_data = *data / l2_norm;
            out_data++;
            data++;
        }
    }
    return 0;
}
