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

int ref_elu_uint8(uint8_t* data, uint8_t* out_data,int size, elu_param* param)
{
    for(int i = 0; i < size; i++)
    {
        float real_data = (data[i]-param->zero_point) * param->scale;
        if(real_data < 0)
        {
            out_data[i] = round((exp(real_data) - 1)/param->scale)+param->zero_point;
        }
        else
        {
            out_data[i] = round(real_data / param->scale)+param->zero_point;
        }
    }
    return 0;
}
