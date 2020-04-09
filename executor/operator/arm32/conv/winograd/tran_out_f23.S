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


//r0: mid
//r1: tmp
//r2: tmp1
//r3: bias
//r4: activation

    .section .text,"ax"
    .align 5

    .type tran_out_f23 STT_FUNC
    .global tran_out_f23
    .hidden tran_out_f23

tran_out_f23:
    vpush       {d8 - d15}
    vmov.i64    d18, #0         //relu
    vmov.i64    d19, #0         //relu
    vmov.i64    d20, #0
    vmov.i64    d21, #0
    cmp         r3,  #0
    beq         no_biases
    vld1.f32    {d20[]},[r3]    // load bias
    vld1.f32    {d21[]},[r3]    // load bias

no_biases:

    ldr         r3, [sp, #0x40]     // r3 activation
    vdup.32     q8, r3
    vcvt.f32.s32    q8, q8

    vldm    r0!, {d0-d3}
    vldm    r0!, {d4-d7}
    vldm    r0!, {d8-d11}
    vldm    r0!, {d12-d15}

    vadd.f32    q12, q0, q4
    vadd.f32    q13, q1, q5
    vldm    r0!, {d0 - d3}
    vadd.f32    q14, q2, q6
    vadd.f32    q15, q3, q7
    vldm    r0!, {d4 - d7}

    vadd.f32    q12, q12, q0
    vadd.f32    q13, q13, q1
    vadd.f32    q14, q14, q2
    vadd.f32    q15, q15, q3

    vsub.f32    q4, q4, q0
    vsub.f32    q5, q5, q1
    vldm    r0!, {d0 - d3}
    vsub.f32    q6, q6, q2
    vsub.f32    q7, q7, q3
    vldm    r0!, {d4 - d7}

    vadd.f32    q4, q4, q0
    vadd.f32    q5, q5, q1
    vadd.f32    q6, q6, q2
    vadd.f32    q7, q7, q3

    vadd.f32    q0, q12, q13
    vsub.f32    q1, q13, q14
    vadd.f32    q0, q0, q14
    vadd.f32    q1, q1, q15

    vadd.f32    q2, q4, q5
    vsub.f32    q3, q5, q6
    vadd.f32    q2, q2, q6
    vadd.f32    q3, q3, q7

    vadd.f32    q0, q0, q10
    vadd.f32    q1, q1, q10
    vadd.f32    q2, q2, q10
    vadd.f32    q3, q3, q10

    cmp     r3, #0
    blt     store
    vmax.f32    q0, q0, q9
    vmax.f32    q1, q1, q9
    vmax.f32    q2, q2, q9
    vmax.f32    q3, q3, q9
    beq     store
    vmin.f32    q0, q0, q8
    vmin.f32    q1, q1, q8
    vmin.f32    q2, q2, q8
    vmin.f32    q3, q3, q8

store:
    vzip.32     q0, q1
    vzip.32     q2, q3
    vstm    r1, {d0-d3}
    vstm    r2, {d4-d7}

    vpop    {d8 - d15}
    bx      lr
    .end

