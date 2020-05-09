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
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/

//
// register definition
// x0        output start address
// x1        input start address
// x2        kernel start address
// x3        in_channel (cin)

    .section .text,"ax"
    .align 5

    .type wino_hgemm_1x16_fp16 STT_FUNC
    .global wino_hgemm_1x16_fp16

wino_hgemm_1x16_fp16:

    cmp    x3, 0x4
	movi	d24, 0x0
    movi	d25, 0x0

    and    x10,x3, 0x3
    b.lt    loop4_end
    lsr    x9, x3, 0x2

loop4:  
    ldr    d0, [x1]                         //input
    ldp    q4, q5, [x2]                    //kernel 
    ldp    q6, q7,   [x2, #0x20]
    ldp    q16, q17, [x2, #0x40]
    ldp    q18, q19, [x2, #0x60]
    subs    x9, x9, 0x1

    fmla    v24.8h, v4.8h,  v0.h[0]
    fmla    v25.8h, v5.8h,  v0.h[0]

    fmla    v24.8h, v6.8h,  v0.h[1]
    fmla    v25.8h, v7.8h,  v0.h[1]

    fmla    v24.8h, v16.8h,  v0.h[2]
    fmla    v25.8h, v17.8h,  v0.h[2]

    fmla    v24.8h, v18.8h,  v0.h[3]
    fmla    v25.8h, v19.8h,  v0.h[3]

    add	x1, x1, 0x8
    add x2, x2, 0x80
    b.ne    loop4


loop4_end:
    cbz    x10, save_result

loop1:
    ldr    s0,[x1],0x2
    ldp    q4, q5, [x2],0x20
    subs    x10, x10 ,0x1
    fmla    v24.8h, v4.8h,  v0.h[0]
    fmla    v25.8h, v5.8h,  v0.h[0]
    b.ne    loop1

save_result:
    stp  q24,q25, [x0]

    ret
        .end
