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
//r4, activation 0x68
//r5, inw   0x6c
//r6, allo_inc  0x70
//r7, real_inc 0x74
//r8, outw  0x78
//r9, outh  0x7c


//d0~v8,  kernel
//d9~17,  input
//d18,    output
//d19,    bias
//d20,    relu 0
//d21,    relu x


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1_nhwc_float
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
    ldr r9, [sp,#0x7c]
    ldr r8, [sp,#0x78]
    sub r9, r9, #2
    sub r8, r8, #2
    cmp r6, #2
    blt END_FUNC
    cmp r3, #0
    beq LOAD_BIAS_FINISH
    vld1.32 {d19}, [r3]
    add r3, r3, #8

LOAD_BIAS_FINISH:
    ldr r5, [sp,#0x6c]
    ldr r7, [sp,#0x74]
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

//block0-1-2 the top line    
//block0 the top left point
    vld1.32 {d9 }, [r12], r11
    vld1.32 {d10}, [r7], r11
    vld1.32 {d12}, [r12], r11
    vld1.32 {d13}, [r7], r11
    
    vmla.f32 d18, d9,  d4
    vmla.f32 d18, d10, d7
    vmla.f32 d18, d12, d5
    vmla.f32 d18, d13, d8

    cmp r3, #0
    beq ADD_BIAS_FINISH_B0
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B0: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B0
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B0
    vmin.f32 d18, d18, d21

RELU_FINISH_B0:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

//block1 the top middle points 
LOOP_W_B1:    
    vld1.32 {d15}, [r12], r11
    vld1.32 {d16}, [r7], r11

    vmla.f32 d18, d9,  d3
    vmla.f32 d18, d10, d6
    vmla.f32 d18, d12, d4
    vmla.f32 d18, d13, d7
    vmla.f32 d18, d15, d5
    vmla.f32 d18, d16, d8

    vmov d9,  d12
    vmov d10, d13
    vmov d12, d15
    vmov d13, d16
//bias
    cmp r3, #0
    beq ADD_BIAS_FINISH_B1
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B1: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B1
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B1
    vmin.f32 d18, d18, d21

RELU_FINISH_B1:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

    sub r8, r8, #1
    cmp r8, #0
    bgt LOOP_W_B1
    
//block2 the top right point
    vmla.f32 d18, d9,  d3
    vmla.f32 d18, d10, d6
    vmla.f32 d18, d12, d4
    vmla.f32 d18, d13, d7

    cmp r3, #0
    beq ADD_BIAS_FINISH_B2
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B2: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B2
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B2
    vmin.f32 d18, d18, d21

RELU_FINISH_B2:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]
   
    mov r14, r7
    mov r7, r12
    sub r12, r12, r10
//block3-4-5 the middle lines
LOOP_H_B345:
//blcok3 the middle left points
    vld1.32 {d9 }, [r12], r11
    vld1.32 {d10}, [r7], r11
    vld1.32 {d11}, [r14], r11
    vld1.32 {d12}, [r12], r11
    vld1.32 {d13}, [r7], r11
    vld1.32 {d14}, [r14], r11
    
    vmla.f32 d18, d9,  d1
    vmla.f32 d18, d10, d4
    vmla.f32 d18, d11, d7
    vmla.f32 d18, d12, d2
    vmla.f32 d18, d13, d5
    vmla.f32 d18, d14, d8
 
    cmp r3, #0
    beq ADD_BIAS_FINISH_B3
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B3: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B3
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B3
    vmin.f32 d18, d18, d21

RELU_FINISH_B3:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]
    
    ldr r8, [sp,#0x78]
    sub r8, r8, #2
//block4 the middle middle points
LOOP_W_B4:
    vld1.32 {d15}, [r12], r11
    vld1.32 {d16}, [r7], r11
    vld1.32 {d17}, [r14], r11

    vmla.f32 d18, d9,  d0
    vmla.f32 d18, d12, d1
    vmla.f32 d18, d15, d2
    vmla.f32 d18, d10, d3
    vmla.f32 d18, d13, d4
    vmla.f32 d18, d16, d5
    vmla.f32 d18, d11, d6
    vmla.f32 d18, d14, d7
    vmla.f32 d18, d17, d8
    
    vmov d9,  d12
    vmov d12, d15
    vmov d10, d13
    vmov d13, d16
    vmov d11, d14
    vmov d14, d17

//bias
    cmp r3, #0
    beq ADD_BIAS_FINISH_B4
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B4: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B4
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B4
    vmin.f32 d18, d18, d21

RELU_FINISH_B4:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

    sub r8, r8, #1
    cmp r8, #0
    bgt LOOP_W_B4

//block5 the middle right points  
    vmla.f32 d18, d9,  d0
    vmla.f32 d18, d10, d3
    vmla.f32 d18, d11, d6
    vmla.f32 d18, d12, d1
    vmla.f32 d18, d13, d4
    vmla.f32 d18, d14, d7
 
    cmp r3, #0
    beq ADD_BIAS_FINISH_B5
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B5: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B5
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B5
    vmin.f32 d18, d18, d21

RELU_FINISH_B5:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]
//
    sub r9, r9, #1
    cmp r9, #0
    bgt LOOP_H_B345
 
//block6-7-8 the bottom line
//block6 the bottom left point
    vld1.32 {d9 }, [r12], r11
    vld1.32 {d10}, [r7], r11
    vld1.32 {d12}, [r12], r11
    vld1.32 {d13}, [r7], r11
    
    vmla.f32 d18, d9,  d1
    vmla.f32 d18, d10, d4
    vmla.f32 d18, d12, d2
    vmla.f32 d18, d13, d5

    cmp r3, #0
    beq ADD_BIAS_FINISH_B6
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B6: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B6
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B6
    vmin.f32 d18, d18, d21

RELU_FINISH_B6:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

    ldr r8, [sp,#0x78]
    sub r8, r8, #2
//block7 the bottom middle points 
LOOP_W_B7:    
    vld1.32 {d15}, [r12], r11
    vld1.32 {d16}, [r7], r11

    vmla.f32 d18, d9,  d0
    vmla.f32 d18, d10, d3
    vmla.f32 d18, d12, d1
    vmla.f32 d18, d13, d4
    vmla.f32 d18, d15, d2
    vmla.f32 d18, d16, d5

    vmov d9,  d12
    vmov d10, d13
    vmov d12, d15
    vmov d13, d16
//bias
    cmp r3, #0
    beq ADD_BIAS_FINISH_B7
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B7: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B7
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B7
    vmin.f32 d18, d18, d21

RELU_FINISH_B7:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]

    sub r8, r8, #1
    cmp r8, #0
    bgt LOOP_W_B7
    
//block8 the bottom right point
    vmla.f32 d18, d9,  d0
    vmla.f32 d18, d10, d3
    vmla.f32 d18, d12, d1
    vmla.f32 d18, d13, d4

    cmp r3, #0
    beq ADD_BIAS_FINISH_B8
    vadd.f32 d18, d18, d19

ADD_BIAS_FINISH_B8: 
//activation
    cmp r4, #0
    blt RELU_FINISH_B8
    vmax.f32 d18, d18, d20
    beq RELU_FINISH_B8
    vmin.f32 d18, d18, d21

RELU_FINISH_B8:     
    vst1.32 {d18}, [r5]
    add r5, r5, r11
   
    vmov.i64 d18, #0
    vdup.f32 d18, d18[0]
//
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
    




