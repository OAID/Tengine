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
 * Author: bingzhang@openailab.com
 */
static int ref_tile_uint8(uint8_t* data, uint8_t* output, std::vector<int> repeat, std::vector<int> inDim,
                          std::vector<int> outDim, float scale, int zero_point)
{
    int index = 0;
    for(int in = 0; in < inDim[0]; in++)
    {
        for(int rn = 0; rn < repeat.at(3); rn++)
        {
            for(int ic = 0; ic < inDim[1]; ic++)
            {
                for(int rc = 0; rc < repeat.at(2); rc++)
                {
                    for(int ih = 0; ih < inDim[2]; ih++)
                    {
                        for(int rh = 0; rh < repeat.at(1); rh++)
                        {
                            for(int iw = 0; iw < inDim[3]; iw++)
                            {
                                for(int rw = 0; rw < repeat.at(0); rw++)
                                {
                                    int inDataSize = in * inDim[1] * inDim[2] * inDim[3] + ic * inDim[2] * inDim[3] +
                                                     ih * inDim[3] + iw;
                                    float real_value = data[inDataSize] * scale;
                                    output[index] = round(real_value / scale) + zero_point;
                                    index++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
