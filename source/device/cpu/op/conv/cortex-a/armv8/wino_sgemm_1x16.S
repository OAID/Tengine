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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/

// register definition
// x0        output start address
// x1        input start address
// x2        kernel start address
// x3        cin

// x9 ~ x10  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_1x16 STT_FUNC
    .global wino_sgemm_1x16
    .hidden wino_sgemm_1x16
    
wino_sgemm_1x16:
    // bring some code ahead to reduce dependency
    prfm    pldl1keep, [x1]
    cmp    x3, 0x4

none_biases:
    movi    d16, 0x0
    movi    d17, 0x0
    movi    d18, 0x0
    movi    d19, 0x0


start:
    and    x10,x3, 0x3
    b.lt    loop4_end
    lsr    x9, x3, 0x2

loop4:  
    ldr    q0, [x1]
    ldp    q4, q5, [x2]
    fmla    v16.4s, v4.4s,  v0.s[0]
    ldp    q6, q7, [x2, 0x20]
    fmla    v17.4s, v5.4s,  v0.s[0]
    subs    x9, x9, 0x1
    ldp    q4, q5, [x2, 0x40]
    fmla    v18.4s, v6.4s,  v0.s[0]
    fmla    v19.4s, v7.4s,  v0.s[0]

    prfm    pldl1keep, [x2, 0x50]
    ldp    q6, q7, [x2, 0x60]    
    fmla    v16.4s, v4.4s,  v0.s[1]
    fmla    v17.4s, v5.4s,  v0.s[1]
    ldp    q4, q5, [x2, 0x80]
    fmla    v18.4s, v6.4s,  v0.s[1]
    fmla    v19.4s, v7.4s,  v0.s[1]
     
    ldp    q6, q7, [x2, 0xa0]    
    fmla    v16.4s, v4.4s,  v0.s[2]
    fmla    v17.4s, v5.4s,  v0.s[2]
    ldp    q4, q5, [x2, 0xc0]
    prfm    pldl1keep, [x2, 0x60]
    fmla    v18.4s, v6.4s,  v0.s[2]
    fmla    v19.4s, v7.4s,  v0.s[2]
    ldp    q6, q7, [x2, 0xe0]
    prfm    pldl1keep, [x1, 0x20]
    add    x1, x1, 0x10
    fmla    v16.4s, v4.4s,  v0.s[3]
    fmla    v17.4s, v5.4s,  v0.s[3]
    add    x2, x2, 0x100
    fmla    v18.4s, v6.4s,  v0.s[3]
    fmla    v19.4s, v7.4s,  v0.s[3]
    b.ne    loop4


loop4_end:
    cbz    x10, save_result

loop1:
    ldr    s0,[x1]
    ldp    q4, q5, [x2]
    fmla    v16.4s, v4.4s,  v0.s[0]
    ldp     q6, q7, [x2, 0x20]    
    fmla    v17.4s, v5.4s,  v0.s[0]
    add     x2, x2, 0x40
    fmla    v18.4s, v6.4s,  v0.s[0]
    add     x1, x1, 0x4
    fmla    v19.4s, v7.4s,  v0.s[0]
    subs    x10, x10 ,0x1
    b.ne    loop1
    
save_result:
    stp  q16,q17, [x0]
    stp  q18,q19, [x0, 0x20]


    ret
        .end

