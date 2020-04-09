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
//x7, rel_inc
//x8, outw
//x9, outh


//v0~v8,  kernel
//v9~17,  input
//v18,    output
//v19,    bias
//v20,    relu 0
//v21,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1_nhwc_float
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
    
     
    movi d20, #0
    dup v20.4s, v20.s[0]
    mov v21.d[0], x4
    dup v21.4s, v21.s[0]
    scvtf v21.4s, v21.4s 
    mov x11, x7
    lsl x11, x11, #2
    mul x10, x5, x11
    
LOOP_C:
    ldr x9, [sp,0x40]
    ldr x8, [sp,0x48]
    sub x9, x9, #2
    sub x8, x8, #2
    cmp x6, #4
    blt END_FUNC
    cmp x3, #0
    movi  d19, #0x0
    beq LOAD_BIAS_FINISH
    ld1 {v19.4s}, [x3], #16

LOAD_BIAS_FINISH:
//kernel coeff, 8 channels as a block, parallel
    //the first 4 channels
    mov x7, x1
    ld1 {v0.4s}, [x7], x11
    ld1 {v1.4s}, [x7], x11
    ld1 {v2.4s}, [x7], x11
    ld1 {v3.4s}, [x7], x11
    ld1 {v4.4s}, [x7], x11
    ld1 {v5.4s}, [x7], x11
    ld1 {v6.4s}, [x7], x11
    ld1 {v7.4s}, [x7], x11
    ld1 {v8.4s}, [x7]

    mov x13, x0
    add x14, x13, x10
    add x15, x14, x10
    
    mov x7, x2

    movi d18, #0
    dup v18.4s, v18.s[0]

//block0-1-2 the top line    
//block0 the top left point
    ld1 {v9.4s }, [x13], x11
    ld1 {v10.4s}, [x14], x11
    ld1 {v12.4s}, [x13], x11
    ld1 {v13.4s}, [x14], x11
    
    fmla v18.4s, v9.4s,  v4.4s
    fmla v18.4s, v10.4s, v7.4s
    fmla v18.4s, v12.4s, v5.4s
    fmla v18.4s, v13.4s, v8.4s

    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B0: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B0
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B0
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B0:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

//block1 the top middle points 
LOOP_W_B1:    
    ld1 {v15.4s}, [x13], x11
    ld1 {v16.4s}, [x14], x11

    fmla v18.4s, v9.4s,  v3.4s
    fmla v18.4s, v10.4s, v6.4s
    fmla v18.4s, v12.4s, v4.4s
    fmla v18.4s, v13.4s, v7.4s
    fmla v18.4s, v15.4s, v5.4s
    fmla v18.4s, v16.4s, v8.4s

    mov v9.16b,  v12.16b
    mov v10.16b, v13.16b
    mov v12.16b, v15.16b
    mov v13.16b, v16.16b
//bias
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B1: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B1
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B1
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B1:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

    sub x8, x8, #1
    cmp x8, #0
    bgt LOOP_W_B1
    
//block2 the top right point
    fmla v18.4s, v9.4s,  v3.4s
    fmla v18.4s, v10.4s, v6.4s
    fmla v18.4s, v12.4s, v4.4s
    fmla v18.4s, v13.4s, v7.4s

    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B2: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B2
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B2
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B2:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]
   
    mov x15, x14
    mov x14, x13
    sub x13, x13, x10
//block3-4-5 the middle lines
LOOP_H_B345:
//blcok3 the middle left points
    ld1 {v9.4s }, [x13], x11
    ld1 {v10.4s}, [x14], x11
    ld1 {v11.4s}, [x15], x11
    ld1 {v12.4s}, [x13], x11
    ld1 {v13.4s}, [x14], x11
    ld1 {v14.4s}, [x15], x11
    
    fmla v18.4s, v9.4s,  v1.4s
    fmla v18.4s, v10.4s, v4.4s
    fmla v18.4s, v11.4s, v7.4s
    fmla v18.4s, v12.4s, v2.4s
    fmla v18.4s, v13.4s, v5.4s
    fmla v18.4s, v14.4s, v8.4s
 
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B3: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B3
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B3
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B3:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]
    
    ldr x8, [sp,0x48]
    sub x8, x8, #2
//block4 the middle middle points
LOOP_W_B4:
    lsl  x12, x11, 0x1
    prfm pldl1keep, [x13, x12]
    prfm pldl1keep, [x14, x12]
    prfm pldl1keep, [x15, x12]
    ld1 {v15.4s}, [x13], x11
    ld1 {v16.4s}, [x14], x11
    ld1 {v17.4s}, [x15], x11

    fmla v18.4s, v9.4s,  v0.4s
    mov v9.16b,  v12.16b
    fmla v18.4s, v12.4s, v1.4s
    mov v12.16b, v15.16b
    fmla v18.4s, v15.4s, v2.4s

    fmla v18.4s, v10.4s, v3.4s
    mov v10.16b, v13.16b
    fmla v18.4s, v13.4s, v4.4s
    mov v13.16b, v16.16b
    fmla v18.4s, v16.4s, v5.4s

    fmla v18.4s, v11.4s, v6.4s
    mov v11.16b, v14.16b
    fmla v18.4s, v14.4s, v7.4s
    mov v14.16b, v17.16b
    fmla v18.4s, v17.4s, v8.4s

//bias
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B4: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B4
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B4
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B4:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

    sub x8, x8, #1
    cmp x8, #0
    bgt LOOP_W_B4

//block5 the middle right points  
    fmla v18.4s, v9.4s,  v0.4s
    fmla v18.4s, v10.4s, v3.4s
    fmla v18.4s, v11.4s, v6.4s
    fmla v18.4s, v12.4s, v1.4s
    fmla v18.4s, v13.4s, v4.4s
    fmla v18.4s, v14.4s, v7.4s
 
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B5: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B5
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B5
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B5:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]
//
    sub x9, x9, #1
    cmp x9, #0
    bgt LOOP_H_B345
 
//block6-7-8 the bottom line
//block6 the bottom left point
    ld1 {v9.4s }, [x13], x11
    ld1 {v10.4s}, [x14], x11
    ld1 {v12.4s}, [x13], x11
    ld1 {v13.4s}, [x14], x11
    
    fmla v18.4s, v9.4s,  v1.4s
    fmla v18.4s, v10.4s, v4.4s
    fmla v18.4s, v12.4s, v2.4s
    fmla v18.4s, v13.4s, v5.4s

    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B6: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B6
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B6
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B6:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

    ldr x8, [sp,0x48]
    sub x8, x8, #2
//block7 the bottom middle points 
LOOP_W_B7:    
    ld1 {v15.4s}, [x13], x11
    ld1 {v16.4s}, [x14], x11

    fmla v18.4s, v9.4s,  v0.4s
    fmla v18.4s, v10.4s, v3.4s
    fmla v18.4s, v12.4s, v1.4s
    fmla v18.4s, v13.4s, v4.4s
    fmla v18.4s, v15.4s, v2.4s
    fmla v18.4s, v16.4s, v5.4s

    mov v9.16b,  v12.16b
    mov v10.16b, v13.16b
    mov v12.16b, v15.16b
    mov v13.16b, v16.16b
//bias
    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B7: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B7
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B7
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B7:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]

    sub x8, x8, #1
    cmp x8, #0
    bgt LOOP_W_B7
    
//block8 the bottom right point
    fmla v18.4s, v9.4s,  v0.4s
    fmla v18.4s, v10.4s, v3.4s
    fmla v18.4s, v12.4s, v1.4s
    fmla v18.4s, v13.4s, v4.4s

    fadd v18.4s, v18.4s, v19.4s

ADD_BIAS_FINISH_B8: 
//activation
    cmp x4, #0
    blt RELU_FINISH_B8
    fmax v18.4s, v18.4s, v20.4s
    beq RELU_FINISH_B8
    fmin v18.4s, v18.4s, v21.4s

RELU_FINISH_B8:     
    st1 {v18.4s}, [x7]
    add x7, x7, x11
   
    movi d18, #0
    dup v18.4s, v18.s[0]
//
    add x0, x0, #16
    add x1, x1, #16
    add x2, x2, #16

    sub x6, x6, #4
    cmp x6, #4
    bge LOOP_C

END_FUNC:
    ldp d8, d9, [sp]
    ldp d10, d11, [sp,0x10] 
    ldp d12, d13, [sp,0x20]
    ldp d14, d15, [sp,0x30]
    add sp, sp, #0x40
    
    ret
    




