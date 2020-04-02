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
//v9~v17, kernel
//v18~20, input
//v21~23, input
//v24,    output
//v25,    output
//v26,    bias
//v27,    bias
//v28,    relu 0
//v29,    relu r


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p0_nhwc_float
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
    push {r4 - r12, lr}
    vpush {d8 - d15}
 
    vmov.i64 d28, #0
    vdup.f32 d28, d28[0]
    ldr r4, [sp,#0x68]
    vmov.32 d29[0], r4
    vdup.f32 d29, d29[0]
    vcvt.f32.s32 d29, d29 
    ldr r6, [sp,#0x70]
 
LOOP_C:
    cmp r6, #4
    blt END_FUNC
    cmp r3, #0
    beq LOAD_BIAS_FINISH
    vld1.32 {d26}, [r3]
    add r3, r3, #8
    vld1.32 {d27}, [r3]
    add r3, r3, #8
    ldr r5, [sp,#0x6c]
    ldr r7, [sp,#0x74]

LOAD_BIAS_FINISH:
//kernel coeff, 8 channels as a block, parallel
    //the first 4 channels
    mov r10, r1
    mov r11, r7
    lsl r11, r11, #2
    vld1.32 {d0}, [r10], r11
    vld1.32 {d1}, [r10], r11
    vld1.32 {d2}, [r10], r11
    vld1.32 {d3}, [r10], r11
    vld1.32 {d4}, [r10], r11
    vld1.32 {d5}, [r10], r11
    vld1.32 {d6}, [r10], r11
    vld1.32 {d7}, [r10], r11
    vld1.32 {d8}, [r10]
    //the second 4 channels
    mov r10, r1
    add r10, r10, #8
    vld1.32 { d9}, [r10], r11
    vld1.32 {d10}, [r10], r11
    vld1.32 {d11}, [r10], r11
    vld1.32 {d12}, [r10], r11
    vld1.32 {d13}, [r10], r11
    vld1.32 {d14}, [r10], r11
    vld1.32 {d15}, [r10], r11
    vld1.32 {d16}, [r10], r11
    vld1.32 {d17}, [r10]

    mul r10, r5, r7
    lsl r10, r10, #2
    mov r12, r0
    add r7, r12, r10
    add r14, r7, r10
    
    mov r5, r2

    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0]

    ldr r9, [sp, #0x7c]

LOOP_H:
//input data, 8 channels as a block, parallel
    //the first 4 channels
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d20}, [r14]
    add r14, r14, #8
    //the second 4 channels
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    vld1.32 {d23}, [r14]
    sub r12, r12, #8
    sub r7, r7, #8
    sub r14, r14, #8
    add r12, r12, r11
    add r7, r7, r11
    add r14, r14, r11

    ldr r8, [sp, #0x78]
 
LOOP_W:    
//compute output data, 8 channels as a block, parallel
    //the first 4 channels
    vmla.f32 d24, d18, d0
    vmla.f32 d24, d19, d3
    vmla.f32 d24, d20, d6
    //the second 4 channels
    vmla.f32 d25, d21,  d9
    vmla.f32 d25, d22, d12
    vmla.f32 d25, d23, d15

//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d20}, [r14]
    add r14, r14, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    vld1.32 {d23}, [r14]
    sub r12, r12, #8
    sub r7, r7, #8
    sub r14, r14, #8
    add r12, r12, r11
    add r7, r7, r11
    add r14, r14, r11
//
    vmla.f32 d24, d18, d1
    vmla.f32 d24, d19, d4
    vmla.f32 d24, d20, d7
    vmla.f32 d25, d21, d10
    vmla.f32 d25, d22, d13
    vmla.f32 d25, d23, d16
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d20}, [r14]
    add r14, r14, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    vld1.32 {d23}, [r14]
    sub r12, r12, #8
    sub r7, r7, #8
    sub r14, r14, #8
    add r12, r12, r11
    add r7, r7, r11
    add r14, r14, r11
//
    vmla.f32 d24, d18, d2
    vmla.f32 d24, d19, d5
    vmla.f32 d24, d20, d8
    vmla.f32 d25, d21, d11
    vmla.f32 d25, d22, d14
    vmla.f32 d25, d23, d17
   
//bias
    cmp r3, #0
    beq ADD_BIAS_FINISH
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH: 
//activation
    cmp r4, #0
    blt RELU_FINISH
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0] 

    sub r8, r8, #1
    cmp r8, #0
    bgt LOOP_W
   
    add r12, r12, r10
    add r7, r7, r10
    add r14, r14, r10
 
    sub r9, r9, #1
    cmp r9, #0
    bgt LOOP_H
    
    add r0, r0, #16
    add r1, r1, #16
    add r2, r2, #16

    sub r6, r6, #4
    cmp r6, #4
    bge LOOP_C

END_FUNC:
    vpop {d8 - d15}
    pop {r4 - r12, pc}

    .end
    




