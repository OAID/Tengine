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
 * Author: chunyinglv@openailab.com
 */

#ifndef __WINO_CONFIG_H__
#define __WINO_CONFIG_H__

#define KER_COUT_UNIT 12
#define KER_COUT_UNIT4 4
#define ELEM_SIZE 36
#define TILE 4
#define BLOCK_HW_UNIT 4

#define WINO_MAX(a, b) ((a) > (b) ? (a) : (b))
#define WINO_MIN(a, b) ((a) < (b) ? (a) : (b))
static inline float do_activation(float input, int activation)
{
    if(activation == 0)
    {
        input = WINO_MAX(input, 0);
        if(activation == 6)
            input = WINO_MIN(input, 6);
    }
    return input;
}
#endif    // __WINO_CONFIG_H__