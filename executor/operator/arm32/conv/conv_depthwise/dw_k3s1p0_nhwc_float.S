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


//r0, input address
//r1, kernel address
//r2, output address
//r3, bias address
//r4, activation
//r5, inw
//r6, allo_inc
//r7, real_inc
//r8, outw
//r9, outh


//d0~8,   kernel
//d9~17,  input
//d18,    output
//d19,    bias
//d20,    relu 0
//d21,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p0_nhwc_float
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
    push {r4 - r12, lr}
    vpush {d8 - d15}
 
    vmov.i64 d20, #0
    vdup.f32 d20, d20[0]
    ldr r4, [sp,#0x68]
    vmov.32 d21[0], r4
    vdup.f32 d21, d21[0] 
    vcvt.f32.s32 d21, d21
    ldr r6, [sp,#0x70]
 
LOOP_C:
    cmp r6, #2
    blt END_FUNC
    cmp r3, #0
    beq LOAD_BIAS_FINISH
    vld1.32 {d19}, [r3]
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

    mul r10, r5, r7
    lsl r10, r10, #2
    mov r12, r0
    add r7, r12, r10
    add r14, r7, r10
    
    mov r5, r2

    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]
    
    ldr r9, [sp, #0x7c]

LOOP_H:
//input data, 8 channels as a block, parallel
    //the first 4 channels
    vld1.32 {d9 }, [r12], r11
    vld1.32 {d10}, [r7], r11
    vld1.32 {d11}, [r14], r11
    vld1.32 {d12}, [r12], r11
    vld1.32 {d13}, [r7], r11
    vld1.32 {d14}, [r14], r11
    
    ldr r8, [sp, #0x78]

LOOP_W:    
//compute output data, 8 channels as a block, parallel
    //the first 4 channels
    vld1.32 {d15}, [r12], r11
    vld1.32 {d16}, [r7], r11
    vld1.32 {d17}, [r14], r11

    vmla.f32 d18, d9,  d0
    vmla.f32 d18, d10, d3
    vmla.f32 d18, d11, d6
    vmla.f32 d18, d12, d1
    vmla.f32 d18, d13, d4
    vmla.f32 d18, d14, d7
    vmla.f32 d18, d15, d2
    vmla.f32 d18, d16, d5
    vmla.f32 d18, d17, d8

    vmov d9,  d12
    vmov d10, d13
    vmov d11, d14
    vmov d12, d15
    vmov d13, d16
    vmov d14, d17
//bias
    cmp r3, #0
    beq ADD_BIAS_FINISH
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH: 
//activation
    cmp r4, #0
    blt RELU_FINISH
    vmax.f32 d18, d18, d20
    beq RELU_FINISH
    vmin.f32 d18, d18, d21

RELU_FINISH:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

    sub r8, r8, #1
    cmp r8, #0
    bgt LOOP_W
   
    sub r9, r9, #1
    cmp r9, #0
    bgt LOOP_H
    
    add r0, r0, #8
    add r1, r1, #8
    add r2, r2, #8

    sub r6, r6, #2
    cmp r6, #2
    bge LOOP_C

END_FUNC:
    vpop {d8 - d15}   
    pop {r4 - r12, pc}

    .end
    




