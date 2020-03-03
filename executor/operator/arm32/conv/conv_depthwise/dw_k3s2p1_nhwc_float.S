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


//d0~v8,  kernel
//d9~v17, kernel
//d18~20, input
//d21~23, input
//d24,    output
//d25,    output
//d26,    bias
//d27,    bias
//d28,    relu 0
//d29,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p1_nhwc_float
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
    ldr r9, [sp,#0x7c]
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

//
//block0-1-2 the top line
//block0 the top left point
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    //the second 4 channels
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d4
    vmla.f32 d24, d19, d7
    vmla.f32 d25, d21, d13
    vmla.f32 d25, d22, d16

    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11
//
    vmla.f32 d24, d18, d5
    vmla.f32 d24, d19, d8
    vmla.f32 d25, d21, d14
    vmla.f32 d25, d22, d17
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_0
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_0: 
//activation
    cmp r4, #0
    blt RELU_FINISH_0
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_0
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_0:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0]

    ldr r8, [sp, #0x78] 
//block1 the top middle points
LOOP_W_B1:    
    vmla.f32 d24, d18, d3
    vmla.f32 d24, d19, d6
    vmla.f32 d25, d21, d12
    vmla.f32 d25, d22, d15

    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d4
    vmla.f32 d24, d19, d7
    vmla.f32 d25, d21, d13
    vmla.f32 d25, d22, d16
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11
//
    vmla.f32 d24, d18, d5
    vmla.f32 d24, d19, d8
    vmla.f32 d25, d21, d14
    vmla.f32 d25, d22, d17
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_1
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_1: 
//activation
    cmp r4, #0
    blt RELU_FINISH_1
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_1
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_1:     
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
    cmp r8, #2
    bgt LOOP_W_B1
//block2 the top right point
    vmla.f32 d24, d18, d3
    vmla.f32 d24, d19, d6
    vmla.f32 d25, d21, d12
    vmla.f32 d25, d22, d15
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d4
    vmla.f32 d24, d19, d7
    vmla.f32 d25, d21, d13
    vmla.f32 d25, d22, d16
    
    cmp r3, #0
    beq ADD_BIAS_FINISH_2
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_2: 
//activation
    cmp r4, #0
    blt RELU_FINISH_2
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_2
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_2:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0] 
    
    add r14, r7, r10
//block3-4-5 the middle lines
LOOP_H_B345:
//block3 the middle left points
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
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_3
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_3: 
//activation
    cmp r4, #0
    blt RELU_FINISH_3
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_3
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_3:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0]
 
    ldr r8, [sp, #0x78]
//block4 the middle middle points    
LOOP_W_B4:    
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
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_4
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_4: 
//activation
    cmp r4, #0
    blt RELU_FINISH_4
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_4
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_4:     
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
    cmp r8, #2
    bgt LOOP_W_B4

//block5 the middle right points
    vmla.f32 d24, d18, d0
    vmla.f32 d24, d19, d3
    vmla.f32 d24, d20, d6
    vmla.f32 d25, d21, d9
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

    vmla.f32 d24, d18, d1
    vmla.f32 d24, d19, d4
    vmla.f32 d24, d20, d7
    vmla.f32 d25, d21, d10
    vmla.f32 d25, d22, d13
    vmla.f32 d25, d23, d16
    
    cmp r3, #0
    beq ADD_BIAS_FINISH_5
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_5: 
//activation
    cmp r4, #0
    blt RELU_FINISH_5
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_5
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_5:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0] 

    add r12, r12, r10
    add r7, r7, r10
    add r14, r14, r10
 
    sub r9, r9, #1
    cmp r9, #2
    bgt LOOP_H_B345
   
//block6-7-8 the bottom line
//block6 the bottom left point
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    //the second 4 channels
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d1
    vmla.f32 d24, d19, d4
    vmla.f32 d25, d21, d10
    vmla.f32 d25, d22, d13
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11
//
    vmla.f32 d24, d18, d2
    vmla.f32 d24, d19, d5
    vmla.f32 d25, d21, d11
    vmla.f32 d25, d22, d14
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_6
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_6: 
//activation
    cmp r4, #0
    blt RELU_FINISH_6
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_6
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_6:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0] 
    
    ldr r8, [sp, #0x78]
//block7 the bottom middle points 
LOOP_W_B7:    
    vmla.f32 d24, d18, d0
    vmla.f32 d24, d19, d3
    vmla.f32 d25, d21, d9
    vmla.f32 d25, d22, d12

    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d1
    vmla.f32 d24, d19, d4
    vmla.f32 d25, d21, d10
    vmla.f32 d25, d22, d13
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11
//
    vmla.f32 d24, d18, d2
    vmla.f32 d24, d19, d5
    vmla.f32 d25, d21, d11
    vmla.f32 d25, d22, d14
   
    cmp r3, #0
    beq ADD_BIAS_FINISH_7
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_7: 
//activation
    cmp r4, #0
    blt RELU_FINISH_7
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_7
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_7:     
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
    cmp r8, #2
    bgt LOOP_W_B7

//block8 the bottom right point
    vmla.f32 d24, d18, d0
    vmla.f32 d24, d19, d3
    vmla.f32 d25, d21, d9
    vmla.f32 d25, d22, d12
//
    vld1.32 {d18}, [r12]
    add r12, r12, #8
    vld1.32 {d19}, [r7]
    add r7, r7, #8
    vld1.32 {d21}, [r12]
    vld1.32 {d22}, [r7]
    sub r12, r12, #8
    sub r7, r7, #8
    add r12, r12, r11
    add r7, r7, r11

    vmla.f32 d24, d18, d1
    vmla.f32 d24, d19, d4
    vmla.f32 d25, d21, d10
    vmla.f32 d25, d22, d13
    
    cmp r3, #0
    beq ADD_BIAS_FINISH_8
    vadd.f32 d24, d24, d26
    vadd.f32 d25, d25, d27

ADD_BIAS_FINISH_8: 
//activation
    cmp r4, #0
    blt RELU_FINISH_8
    vmax.f32 d24, d24, d28
    vmax.f32 d25, d25, d28
    beq RELU_FINISH_8
    vmin.f32 d24, d24, d29
    vmin.f32 d25, d25, d29

RELU_FINISH_8:     
    vst1.32 {d24}, [r5]
    add r5, r5, #8
    vst1.32 {d25}, [r5]
    sub r5, r5, #8
    add r5, r5, r11
   
    vmov.i64 d24, #0
    vdup.f32 d24, d24[0]
    vmov.i64 d25, #0
    vdup.f32 d25, d25[0] 
 
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




