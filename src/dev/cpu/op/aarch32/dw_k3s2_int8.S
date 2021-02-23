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
 * Author: haoluo@openailab.com
 */
//      x0: input data  (int 8)
//      x1: kernel data (int 8)
//      x2: output      (int 8)
//      x3: bias data (int 32)
//      x4: out_h
//      x5: out_w
//      x6: multi
//      x7: shift
//      sp: input_w
//      sp + 8: act_min
//      sp + 16: act_max

// input:
//              0  1  2  3  4  5  6  7  8  9
//      line0:  a0 b0 c0 d0 e0 f0 g0 h0 i0 j0
//      line1:  a1 b1 c1 d1 e1 f1 g1 h1 i1 j1
//      line2:  a2 b2 c2 d2 e2 f2 g2 h2 i2 j2
// weight:
//      v20：k00  k01  k02   0   k00   k01   k02   0
//      v21： 0   k00  k01  k02   0    k00   k01  k02
//      v22：k10  k11  k12   0   k10   k11   k12   0
//      v23： 0   k10  k11  k12   0    k10   k11  k12
//      v24：k20  k21  k22   0   k20   k21   k22   0
//      v25： 0   k20  k21  k22   0    k20   k21   k22

// line 0 : v0,v1
// line 1 : v2,v3
// line 2 : v4,v5
// out_scale: v6
// int16x8: v7  ~ v10
// int16x8: v11 ~ v14
// int32x4: v16 ~ v19
// kernel : v20 ~ v25
// out    : v2,v3
// tmp    : v26 v28
// bias   : v27
// relu value: v29

// line 0:-----------smull-------------> int16x8
//      v7 v8 v9 v10
// line 1:-----------smlal-------------> int16x8
//      v7 v8 v9 v10
//  ------------------sadalp-----------> int32x4
//      v16 v17 v18 v19 = saddlp(v7 v8 v9 v10)
// line 2:
//  ------------------smull------------> int16x8
//      v11 v12 v14 v15
//  -----------------sadalp------------> int32x4
//      v16 v17 v18 v19 = sadalp(v11 v12 v14 v15)

// out:
//    v0.4s = addp v7 v8
//    v1.4s = addp v9 v10
// =======================for each line ============================
// step1:
//      a0    b0   c0   d0   e0   f0   g0   h0     ----> out0   out4
//      k00  k01  k02   0   k00   k01  k02  0
//      a0    b0  c0    d0   e0   f0   g0   h0     ----> out1   out5
//      0    k00  k01   k02  0    k00  k01  k02
// step1: extern
//      c0   d0   e0   f0   g0    h0   i0   j0     ----> out2   out6
//      k00  k01  k02   0   k00   k01  k02   0
//      c0   d0    e0    f0  g0   h0   i0   j0     ----> out3   out7
//      0    k00  k01   k02  0    k00  k01  k02
#ifndef KERNEL_NAME
#define KERNEL_NAME depthwise_k3s2_int8
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
    push        {r4 - r12}
    vpush       {d8 - d15}
    
    ldr         r9, [sp, #0x74]     //input_w_pad
    ldr         r4, [sp, #0x64]
    ldr         r5, [sp, #0x68]

    //clear the weight vector q0-q2
    vmov.i64 d0,#0
    vmov.i64 d1,#0
    vmov.i64 d2,#0
    vmov.i64 d3,#0
    vmov.i64 d4,#0
    vmov.i64 d5,#0

    vmov.i64 d6,#0

    //init the weight data
    ldr r12, [r1]
    lsl r12, #0x8
    lsr r12, #0x8
    vdup.s32 d0, r12
    ldr r12, [r1, #3]
    lsl r12, #0x8
    lsr r12, #0x8
    vdup.s32 d1, r12
    ldr r12, [r1, #6]
    lsl r12, #0x8
    lsr r12, #0x8
    vdup.s32 d2, r12

loop_h:
    mov     r10, r0
    add     r11, r0, r9               // input line1
    add     r12, r0, r9, lsl #1       // input line2
    lsr     r6, r5, #0x2
    cmp     r6,#0x0
    beq     loop_w_4

loop_w_8:
    //vldm     r10, {d6 - d7}
    //vldm     r11, {d8 - d9}
    //vldm     r12, {d10 - d11}
    vld1.8     {d6, d7}, [r10]
    vld1.8     {d8, d9}, [r11]
    vld1.8     {d10, d11}, [r12]
    vext.8     q9, q3, q3, #2
    vext.8     q10, q4, q4, #2
    vext.8     q11, q5, q5, #2
    vmull.s8   q12, d6, d0       //0  2
    vmull.s8   q13, d18, d0       //1  3
    
    vmlal.s8   q12, d8, d1
    vmlal.s8   q13, d20, d1
    vpaddl.s16  q12, q12 
    vpaddl.s16  q13, q13
    
    vmull.s8   q3, d10, d2
    vmull.s8   q4, d22, d2
    vpadal.s16  q12, q3
    vpadal.s16  q13, q4

    vpadd.s32   d24, d24, d25    // out0 out2
    vpadd.s32   d25, d26, d27    // out1 out3
    
    vtrn.32     d24, d25     // out0 out1 out2 out3
    
    vmov.i64 d12, #0x0
    vmov.s32 d13, d12
    cmp     r3, #0x0
    beq     no_bias
    ldr     r8, [r3]
    vdup.s32 q6, r8

no_bias:
    vadd.s32     q12,  q12,  q6

    ldr             r7, [sp, #0x6c]
    vdup.s32        q6, r7          //mutli
    VQRDMULH.s32    q12, q12, q6
//debug
    //mov             r7, #-5
    ldr             r7, [sp, #0x70]
    vdup.s32        q6, r7          //shift
    mov             r7, #0x0
    vdup.s32        q7, r7
    vmax.s32        q8, q6, q7
    vmin.s32        q7, q6, q7

    vshl.s32        q12, q12, q8
    ldr             r8, [sp,#0x7c]
    vdup.s32        q8, r8
    vrshl.s32       q12, q12, q7
    ldr             r8, [sp,#0x78]
    vdup.s32        q7, r8
    
    vmax.s32        q12, q12, q7
    vmin.s32        q12, q12, q8

save_w_8:
    vst1.8       d24[0], [r2]!
    vst1.8       d24[4], [r2]!
    vst1.8       d25[0], [r2]!
    vst1.8       d25[4], [r2]!
    //add r2, #1

    add     r10, r10, #0x8
    add     r11, r11, #0x8
    add     r12, r12, #0x8
    subs    r6, r6, #0x1
    bne     loop_w_8

loop_w_4:
    and     r6, r5, #0x3
    cmp     r6, #0x0
    beq     loop_h_end

    //vldm     r10, {d6 - d7}
    //vldm     r11, {d8 - d9}
    //vldm     r12, {d10 - d11}
    vld1.8     {d6, d7}, [r10]
    vld1.8     {d8, d9}, [r11]
    vld1.8     {d10, d11}, [r12]
    vext.8     q9, q3, q3, #2
    vext.8     q10, q4, q4, #2
    vext.8     q11, q5, q5, #2
    vmull.s8   q12, d6, d0       //0  2
    vmull.s8   q13, d18, d0       //1  3
    
    vmlal.s8   q12, d8, d1
    vmlal.s8   q13, d20, d1
    vpaddl.s16  q12, q12 
    vpaddl.s16  q13, q13
    
    vmull.s8   q3, d10, d2
    vmull.s8   q4, d22, d2
    vpadal.s16  q12, q3
    vpadal.s16  q13, q4

    vpadd.s32   d24, d24, d25    // out0 out2
    vpadd.s32   d25, d26, d27    // out1 out3
    
    vtrn.32     d24, d25     // out0 out1 out2 out3
    
    vmov.i64 d12, #0x0
    vmov.s32 d13, d12
    cmp     r3, #0x0
    beq     no_bias1
    ldr     r8, [r3]
    vdup.s32 q6, r8

no_bias1:
    vadd.s32     q12,  q12,  q6

    ldr             r7, [sp, #0x6c]
    vdup.s32        q6, r7          //mutli
    VQRDMULH.s32    q12, q12, q6
//debug
    //mov             r7, #-5
    ldr             r7, [sp, #0x70]
    vdup.s32        q6, r7          //shift
    mov             r7, #0x0
    vdup.s32        q7, r7
    vmax.s32        q8, q6, q7
    vmin.s32        q7, q6, q7

    vshl.s32        q12, q12, q8
    ldr             r8, [sp,#0x7c]
    vdup.s32        q8, r8
    vrshl.s32       q12, q12, q7
    ldr             r8, [sp,#0x78]
    vdup.s32        q7, r8
    
    vmax.s32        q12, q12, q7
    vmin.s32        q12, q12, q8

save_w_1:
    vst1.8      d24[0], [r2]!
    vext.8     q12, q12, q12, #4
    subs    r6, r6, #0x1
    bne     save_w_1

loop_h_end:
    subs    r4, r4, #0x1
    add     r0, r0, r9, lsl #1
    bne     loop_h

loop_end:
    vpop {d8 - d15}
    pop  {r4 - r12}
    bx lr
    .end
