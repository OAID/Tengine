/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or More contributor license agreements.  See the NOTICE file
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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xiaowei@openailab.com
 */
//
// depth wise convolution kernel size 3x3 stride 2  pad 1
//
// input:
//         r0     arg0  input data address 
//         r1     arg1  kernel data address
//         r2     arg2  output data address
//         r3     arg3  channel number
//         sp     arg4  input width
//         sp+0x4 arg5  input height
//         sp+0x8 arg6  bias data address
// output: no
//     

#define KERNEL_NAME dw_k3s2_relu_fused
#define CONV_RELU_FUSE

#include "./dw_k3s2.S"
