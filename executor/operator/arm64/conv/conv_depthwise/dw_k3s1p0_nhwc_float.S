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
 * Author: Renzun
 */


//x0, input address
//x1, kernel address
//x2, output address
//x3, bias address
//x4, activation
//x5, inw
//x6, allo_inc
//x7, real_inc
//x8, outw
//x9, outh


//v0~v8,  kernel
//v9~17,  input
//v18,    output
//v19,    bias
//v20,    relu 0
//v21,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p0_nhwc_float
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
    sub sp, sp, #0x40
    stp d8, d9, [sp]
    stp d10, d11, [sp,0x10]
    stp d12, d13, [sp,0x20]
    stp d14, d15, [sp,0x30]
     
    mov x19, x6
    movi d20, #0
    dup v20.4s, v20.s[0]
    ins v21.d[0], x4
    dup v21.4s, v21.s[0] 
    scvtf v21.4s, v21.4s
 
LOOP_C:
    cmp x19, #4
    blt END_FUNC
    cmp x3, #0
    beq LOAD_BIAS_FINISH
    ld1 {v19.4s}, [x3], #16

LOAD_BIAS_FINISH:
//kernel coeff, 8 channels as a block, parallel
    //the first 4 channels
    mov x10, x1
    mov x11, x7
    lsl x11, x11, #2
    ld1 {v0.4s}, [x10], x11
    ld1 {v1.4s}, [x10], x11
    ld1 {v2.4s}, [x10], x11
    ld1 {v3.4s}, [x10], x11
    ld1 {v4.4s}, [x10], x11
    ld1 {v5.4s}, [x10], x11
    ld1 {v6.4s}, [x10], x11
    ld1 {v7.4s}, [x10], x11
    ld1 {v8.4s}, [x10]

    mul x10, x5, x7
    lsl x10, x10, #2
    mov x16, x0
    add x17, x16, x10
    add x18, x17, x10
    
    mov x20, x2

    movi d18, #0
    dup v18.4s, v18.s[0]
    
    ldr x9, [sp, 0x40]

LOOP_H:
//input data, 8 channels as a block, parallel
    //the first 4 channels
    ld1 {v9.4s }, [x16], x11
    ld1 {v10.4s}, [x17], x11
    ld1 {v11.4s}, [x18], x11
    ld1 {v12.4s}, [x16], x11
    ld1 {v13.4s}, [x17], x11
    ld1 {v14.4s}, [x18], x11
    
    ldr x8, [sp, 0x48]

LOOP_W:    
//compute output data, 8 channels as a block, parallel
    //the first 4 channels
    ld1 {v15.4s}, [x16], x11
    ld1 {v16.4s}, [x17], x11
    ld1 {v17.4s}, [x18], x11

    fmla v18.4s, v9.4s,  v0.4s
    fmla v18.4s, v10.4s, v3.4s
    fmla v18.4s, v11.4s, v6.4s
    fmla v18.4s, v12.4s, v1.4s
    fmla v18.4s, v13.4s, v4.4s
    fmla v18.4s, v14.4s, v7.4s
    fmla v18.4s, v15.4s, v2.4s
    fmla v18.4s, v16.4s, v5.4s
    fmla v18.4s, v17.4s, v8.4s

    mov v9.8b,  v12.8b
    mov v10.8b, v13.8b
    mov v11.8b, v14.8b
    mov v12.8b, v15.8b
    mov v13.8b, v16.8b
    mov v14.8b, v17.8b
//bias
    cmp x3, #0
    beq ADD_BIAS_FINISH
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH: 
//activation
    cmp x4, #0
    blt RELU_FINISH
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH:     
    st1 {v18.4s}, [x20]
    add x20, x20, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

    sub x8, x8, #1
    cmp x8, #0
    bgt LOOP_W
   
    sub x9, x9, #1
    cmp x9, #0
    bgt LOOP_H
    
    add x0, x0, #16
    add x1, x1, #16
    add x2, x2, #16

    sub x19, x19, #4
    cmp x19, #4
    bge LOOP_C

END_FUNC:
    ldp d8, d9, [sp]
    ldp d10, d11, [sp,0x10] 
    ldp d12, d13, [sp,0x20]
    ldp d14, d15, [sp,0x30]
    add sp, sp, #0x40
    
    ret
    




