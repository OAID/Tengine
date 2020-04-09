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
//x6, inc
//x7, rel_c
//x8, outw
//x9, outh


//v0~v8,  kernel
//v9~v17, kernel
//v18~20, input
//v21~23, input
//v24,    output
//v25,    output
//v26,    bias
//v27,    bias
//v28,    relu 0
//v29,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p0p1_nhwc_float
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
     
    movi d28, #0
    dup v28.4s, v28.s[0]
    ins v29.d[0], x4
    dup v29.4s, v29.s[0]
    scvtf v29.4s, v29.4s
    
LOOP_C:
    ldr x9, [sp, 0x40]
    cmp x6, #8
    blt END_FUNC
    cmp x3, #0
    beq LOAD_BIAS_FINISH
    ld1 {v26.4s}, [x3], #16
    ld1 {v27.4s}, [x3], #16

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
    //the second 4 channels
    mov x10, x1
    add x10, x10, #16
    ld1 { v9.4s}, [x10], x11
    ld1 {v10.4s}, [x10], x11
    ld1 {v11.4s}, [x10], x11
    ld1 {v12.4s}, [x10], x11
    ld1 {v13.4s}, [x10], x11
    ld1 {v14.4s}, [x10], x11
    ld1 {v15.4s}, [x10], x11
    ld1 {v16.4s}, [x10], x11
    ld1 {v17.4s}, [x10]

    mul x10, x5, x7
    lsl x10, x10, #2
    mov x12, x0
    add x13, x12, x10
    add x14, x13, x10
    
    mov x15, x2

    movi d24, #0
    dup v24.4s, v24.s[0]
    movi d25, #0
    dup v25.4s, v25.s[0]
//the lines and columns that no need to pad start ---block0
LOOP_H_B01:
    ldr x8, [sp, 0x48]
//input data, 8 channels as a block, parallel
    //the first 4 channels
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v20.4s}, [x14], #16
    //the second 4 channels
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    ld1 {v23.4s}, [x14]
    sub x12, x12, #16
    sub x13, x13, #16
    sub x14, x14, #16
    add x12, x12, x11
    add x13, x13, x11
    add x14, x14, x11
 
LOOP_W_B0:    
//compute output data, 8 channels as a block, parallel
    //the first 4 channels
    fmla v24.4s, v18.4s, v0.4s
    fmla v24.4s, v19.4s, v3.4s
    fmla v24.4s, v20.4s, v6.4s
    //the second 4 channels
    fmla v25.4s, v21.4s,  v9.4s
    fmla v25.4s, v22.4s, v12.4s
    fmla v25.4s, v23.4s, v15.4s

//
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v20.4s}, [x14], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    ld1 {v23.4s}, [x14]
    sub x12, x12, #16
    sub x13, x13, #16
    sub x14, x14, #16
    add x12, x12, x11
    add x13, x13, x11
    add x14, x14, x11

    fmla v24.4s, v18.4s, v1.4s
    fmla v24.4s, v19.4s, v4.4s
    fmla v24.4s, v20.4s, v7.4s
    fmla v25.4s, v21.4s, v10.4s
    fmla v25.4s, v22.4s, v13.4s
    fmla v25.4s, v23.4s, v16.4s
//
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v20.4s}, [x14], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    ld1 {v23.4s}, [x14]
    sub x12, x12, #16
    sub x13, x13, #16
    sub x14, x14, #16
    add x12, x12, x11
    add x13, x13, x11
    add x14, x14, x11
//
    fmla v24.4s, v18.4s, v2.4s
    fmla v24.4s, v19.4s, v5.4s
    fmla v24.4s, v20.4s, v8.4s
    fmla v25.4s, v21.4s, v11.4s
    fmla v25.4s, v22.4s, v14.4s
    fmla v25.4s, v23.4s, v17.4s
   
    cmp x3, #0
    beq ADD_BIAS_FINISH_0
    fadd v24.4s, v24.4s, v26.4s
    fadd v25.4s, v25.4s, v27.4s

ADD_BIAS_FINISH_0: 
//activation
    cmp x4, #0
    blt RELU_FINISH_0
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    beq RELU_FINISH_0
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s

RELU_FINISH_0:     
    stp q24, q25, [x15]
    add x15, x15, x11
   
    movi d24, #0
    dup v24.4s, v24.s[0]
    movi d25, #0
    dup v25.4s, v25.s[0] 

    sub x8, x8, #1
    cmp x8, #1
    bgt LOOP_W_B0
//the lines and columns that no need to pad end ---block0
//the pad column start ---block1
    fmla v24.4s, v18.4s, v0.4s
    fmla v24.4s, v19.4s, v3.4s
    fmla v24.4s, v20.4s, v6.4s
    fmla v25.4s, v21.4s,  v9.4s
    fmla v25.4s, v22.4s, v12.4s
    fmla v25.4s, v23.4s, v15.4s
//
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v20.4s}, [x14], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    ld1 {v23.4s}, [x14]
    sub x12, x12, #16
    sub x13, x13, #16
    sub x14, x14, #16
    add x12, x12, x11
    add x13, x13, x11
    add x14, x14, x11

    fmla v24.4s, v18.4s, v1.4s
    fmla v24.4s, v19.4s, v4.4s
    fmla v24.4s, v20.4s, v7.4s
    fmla v25.4s, v21.4s, v10.4s
    fmla v25.4s, v22.4s, v13.4s
    fmla v25.4s, v23.4s, v16.4s
    
    cmp x3, #0
    beq ADD_BIAS_FINISH_1
    fadd v24.4s, v24.4s, v26.4s
    fadd v25.4s, v25.4s, v27.4s

ADD_BIAS_FINISH_1: 
//activation
    cmp x4, #0
    blt RELU_FINISH_1
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    beq RELU_FINISH_1
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s

RELU_FINISH_1:     
    stp q24, q25, [x15]
    add x15, x15, x11
   
    movi d24, #0
    dup v24.4s, v24.s[0]
    movi d25, #0
    dup v25.4s, v25.s[0] 
//the pad column end ---block1

    add x12, x12, x10
    add x13, x13, x10
    add x14, x14, x10
 
    sub x9, x9, #1
    cmp x9, #1
    bgt LOOP_H_B01
   
//the pad line start ---block2
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    //the second 4 channels
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    sub x12, x12, #16
    sub x13, x13, #16
    add x12, x12, x11
    add x13, x13, x11
    ldr x8, [sp, 0x48]
 
LOOP_W_B2:    
    fmla v24.4s, v18.4s, v0.4s
    fmla v24.4s, v19.4s, v3.4s
    fmla v25.4s, v21.4s,  v9.4s
    fmla v25.4s, v22.4s, v12.4s

    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    sub x12, x12, #16
    sub x13, x13, #16
    add x12, x12, x11
    add x13, x13, x11

    fmla v24.4s, v18.4s, v1.4s
    fmla v24.4s, v19.4s, v4.4s
    fmla v25.4s, v21.4s, v10.4s
    fmla v25.4s, v22.4s, v13.4s
//
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    sub x12, x12, #16
    sub x13, x13, #16
    add x12, x12, x11
    add x13, x13, x11
//
    fmla v24.4s, v18.4s, v2.4s
    fmla v24.4s, v19.4s, v5.4s
    fmla v25.4s, v21.4s, v11.4s
    fmla v25.4s, v22.4s, v14.4s
   
    cmp x3, #0
    beq ADD_BIAS_FINISH_2
    fadd v24.4s, v24.4s, v26.4s
    fadd v25.4s, v25.4s, v27.4s

ADD_BIAS_FINISH_2: 
//activation
    cmp x4, #0
    blt RELU_FINISH_2
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    beq RELU_FINISH_2
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s

RELU_FINISH_2:     
    st1 {v24.4s}, [x15], #16
    st1 {v25.4s}, [x15]
    sub x15, x15, #16
    add x15, x15, x11
   
    movi d24, #0
    dup v24.4s, v24.s[0]
    movi d25, #0
    dup v25.4s, v25.s[0] 

    sub x8, x8, #1
    cmp x8, #1
    bgt LOOP_W_B2
//the pad line end ---block2
//the pad column start ---block3
    fmla v24.4s, v18.4s, v0.4s
    fmla v24.4s, v19.4s, v3.4s
    fmla v25.4s, v21.4s,  v9.4s
    fmla v25.4s, v22.4s, v12.4s
//
    ld1 {v18.4s}, [x12], #16
    ld1 {v19.4s}, [x13], #16
    ld1 {v21.4s}, [x12]
    ld1 {v22.4s}, [x13]
    sub x12, x12, #16
    sub x13, x13, #16
    add x12, x12, x11
    add x13, x13, x11

    fmla v24.4s, v18.4s, v1.4s
    fmla v24.4s, v19.4s, v4.4s
    fmla v25.4s, v21.4s, v10.4s
    fmla v25.4s, v22.4s, v13.4s
    
    cmp x3, #0
    beq ADD_BIAS_FINISH_3
    fadd v24.4s, v24.4s, v26.4s
    fadd v25.4s, v25.4s, v27.4s

ADD_BIAS_FINISH_3: 
//activation
    cmp x4, #0
    blt RELU_FINISH_3
    fmax v24.4s, v24.4s, v28.4s
    fmax v25.4s, v25.4s, v28.4s
    beq RELU_FINISH_3
    fmin v24.4s, v24.4s, v29.4s
    fmin v25.4s, v25.4s, v29.4s

RELU_FINISH_3:     
    st1 {v24.4s}, [x15], #16
    st1 {v25.4s}, [x15]
    sub x15, x15, #16
    add x15, x15, x11
   
    movi d24, #0
    dup v24.4s, v24.s[0]
    movi d25, #0
    dup v25.4s, v25.s[0] 
//the pad column end ---block3
 
    add x0, x0, #32
    add x1, x1, #32
    add x2, x2, #32

    sub x6, x6, #8
    cmp x6, #8
    bge LOOP_C

END_FUNC:
    ldp d8, d9, [sp]
    ldp d10, d11, [sp,0x10] 
    ldp d12, d13, [sp,0x20]
    ldp d14, d15, [sp,0x30]
    add sp, sp, #0x40
    
    ret
    




