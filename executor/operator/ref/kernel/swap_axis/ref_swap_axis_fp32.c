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
 * Author: haoluo@openailab.com
 */

static int ref_swap_axis_fp32(const float* in_data,float* out_data,const int* dims)
{
    
    for(int i = 0; i < dims[0]; i ++)
        for(int j = 0; j < dims[3]; j ++)
            for(int p = 0; p < dims[2]; p ++)
                for(int q = 0; q < dims[1]; q ++)
                {
                    int out_index = i*dims[1]*dims[2]*dims[3]*dims[4] + j*dims[2]*dims[1]*dims[4]
                                    + p*dims[1]*dims[4] + q*dims[4];
                    int in_index = i*dims[1]*dims[2]*dims[3]*dims[4] + q*dims[2]*dims[3]*dims[4]
                                    + p*dims[3]*dims[4] + j*dims[4];
                    memcpy(out_data + out_index, in_data + in_index, dims[4]*sizeof(float));
                }

    return 0;
}
