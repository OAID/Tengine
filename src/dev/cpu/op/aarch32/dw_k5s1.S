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
 * Author: haoluo@openailab.com
 */

//
// depthwise convolution kernel size 5x5 stride 1
// input:
//        r0     arg0  input data address
//        r1     arg1  kernel data address
//        r2     arg3  bias
//        r3     arg2  output data address
//        r4 #0x68     arg4  output h
//        r5 #0x6c     arg5  output w
//        r6 #0x70     arg6  activation
// output: no
//
// kernel d0-d12 0-26
//
// input  q12 ~ q13
//
// bias    q15
// relu0   q14
// relu6   q7
//
// output q10,q11
//
//


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k5s1
#endif

    .section .text, "ax"
    .align 5
    .type KERNEL_NAME STT_FUNC
    .global KERNEL_NAME
    .hidden KERNEL_NAME
KERNEL_NAME:
	push		{r4 - r12, lr}
	vpush		{d8 - d15}
    // context save & load parameter
	ldr		r12, [sp, #0x70]		// r12 activation
	ldr		r4,  [sp, #0x68]		// r4 output_h
	ldr		r5,  [sp, #0x6c]		// r5 output_w
	vmov.i64	d28, #0			// for relu
	vmov.i64	d29, #0			// for relu
    vdup.32         q7, r12
    vcvt.f32.s32    q7, q7

	vmov.i64	d30, #0			// bias
	vmov.i64	d31, #0			// bias
    teq             r2 ,#0x0
    beq             no_biases
    vld1.f32        {d30[]},[r2]   	// load bias
    vld1.f32        {d31[]},[r2]   	// load bias
no_biases:

	vldm		r1!, {d0 - d12}
    add     r1, r5, #0x4
    cmp     r4, #0x2
    lsl     r1, r1, #0x2
    blt     loop_h1

loop_h2:

    add     r6, r0, r1
    add     r7, r0, r1, lsl #0x1
    add     r8, r6, r1, lsl #0x1
    add     r9, r0, r1, lsl #0x2
    add     r10, r6, r1, lsl #0x2
    add     r2, r3, r5, lsl #0x2
    lsr     r11, r5, #0x2
    cmp     r11, #0
    beq     loop_h2_w1

loop_h2_w4:
    vldm        r0, {d24-d27}
    add         r0, r0, #0x10
    vext.32     q9, q12, q13, #1
    vmul.f32    q10, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q13, d2[0]
    vmla.f32    q10, q8, d1[0]
    vldm        r6, {d24-d27}
    vmla.f32    q10, q9, d1[1]

    add         r6, r6, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d2[1]
    vmul.f32    q11, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d3[0]
    vmla.f32    q11, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q13, d4[1]
    vmla.f32    q11, q13, d2[0]
    vmla.f32    q10, q8, d3[1]
    vmla.f32    q11, q8, d1[0]
    vldm        r7, {d24-d27}
    vmla.f32    q10, q9, d4[0]
    vmla.f32    q11, q9, d1[1]

    add         r7, r7, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d5[0]
    vmla.f32    q11, q12, d2[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d5[1]
    vmla.f32    q11, q9, d3[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d6[0]
    vmla.f32    q11, q8, d3[1]
    vmla.f32    q10, q13, d7[0]
    vmla.f32    q11, q13, d4[1]
    vldm        r8, {d24-d27}
    vmla.f32    q10, q9, d6[1]
    vmla.f32    q11, q9, d4[0]

    add         r8, r8, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d7[1]
    vmla.f32    q11, q12, d5[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d8[0]
    vmla.f32    q11, q9, d5[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d8[1]
    vmla.f32    q11, q8, d6[0]
    vmla.f32    q10, q13, d9[1]
    vmla.f32    q11, q13, d7[0]
    vldm        r9, {d24-d27}
    vmla.f32    q10, q9, d9[0]
    vmla.f32    q11, q9, d6[1]

    add         r9, r9, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d10[0]
    vmla.f32    q11, q12, d7[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d10[1]
    vmla.f32    q11, q9, d8[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d11[0]
    vmla.f32    q11, q8, d8[1]
    vmla.f32    q10, q13, d12[0]
    vmla.f32    q11, q13, d9[1]
    vldm        r10, {d24-d27}
    vmla.f32    q10, q9, d11[1]
    vmla.f32    q11, q9, d9[0]

    add         r10, r10, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q11, q12, d10[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q11, q9, d10[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q11, q8, d11[0]
    vmla.f32    q11, q9, d11[1]
    vadd.f32    q10, q10, q15
    vmla.f32    q11, q13, d12[0]
    vadd.f32    q11, q11, q15
    cmp         r12, #0

    blt         loop_h2_w4_save
	vmax.f32	q10, q10, q14
	vmax.f32	q11, q11, q14
    beq         loop_h2_w4_save
	vmin.f32	q10, q10, q7
	vmin.f32	q11, q11, q7
loop_h2_w4_save:
    vstm    r3!, {d20-d21}
    vstm    r2!, {d22-d23}
    subs    r11, r11, #0x1
    bne     loop_h2_w4

loop_h2_w1:
    and     r11, r5, #0x3
    cmp     r11, #0
    beq     loop_h2_end

    vldm        r0, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmul.f32    q10, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d1[0]
    vmla.f32    q10, q13, d2[0]
    vldm        r6, {d24-d27}
    vmla.f32    q10, q9, d1[1]

    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d2[1]
    vmul.f32    q11, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d3[0]
    vmla.f32    q11, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d3[1]
    vmla.f32    q11, q8, d1[0]
    vmla.f32    q10, q13, d4[1]
    vmla.f32    q11, q13, d2[0]
    vldm        r7, {d24-d27}
    vmla.f32    q10, q9, d4[0]
    vmla.f32    q11, q9, d1[1]

    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d5[0]
    vmla.f32    q11, q12, d2[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d5[1]
    vmla.f32    q11, q9, d3[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d6[0]
    vmla.f32    q11, q8, d3[1]
    vmla.f32    q10, q13, d7[0]
    vmla.f32    q11, q13, d4[1]
    vldm        r8, {d24-d27}
    vmla.f32    q10, q9, d6[1]
    vmla.f32    q11, q9, d4[0]

    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d7[1]
    vmla.f32    q11, q12, d5[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d8[0]
    vmla.f32    q11, q9, d5[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d8[1]
    vmla.f32    q11, q8, d6[0]
    vmla.f32    q10, q13, d9[1]
    vmla.f32    q11, q13, d7[0]
    vldm        r9, {d24-d27}
    vmla.f32    q10, q9, d9[0]
    vmla.f32    q11, q9, d6[1]

    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d10[0]
    vmla.f32    q11, q12, d7[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d10[1]
    vmla.f32    q11, q9, d8[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d11[0]
    vmla.f32    q11, q8, d8[1]
    vmla.f32    q10, q13, d12[0]
    vmla.f32    q11, q13, d9[1]
    vldm        r10, {d24-d27}
    vmla.f32    q10, q9, d11[1]
    vmla.f32    q11, q9, d9[0]

    vext.32     q9, q12, q13, #1
    vmla.f32    q11, q12, d10[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q11, q9, d10[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q11, q8, d11[0]
    vadd.f32    q10, q10, q15
    vmla.f32    q11, q9, d11[1]
    add         r0, r0, r11, lsl #2
    vmla.f32    q11, q13, d12[0]
    vadd.f32    q11, q11, q15
    cmp         r12, #0

    blt         loop_h2_w1_save
	vmax.f32	q10, q10, q14
	vmax.f32	q11, q11, q14
    beq         loop_h2_w1_save
	vmin.f32	q10, q10, q7
	vmin.f32	q11, q11, q7

loop_h2_w1_save:
    vmov.f32    d13, d20
    vext.32     q10, q10, q10, #1
    vstr        s26, [r3]
    add         r3, r3, #0x4
    vmov.f32    d13, d22
    vext.32     q11, q11, q11, #1
    vstr        s26, [r2]
    add         r2, r2, #0x4
    subs        r11, r11, #0x1
    bne         loop_h2_w1_save

loop_h2_end:
    add     r0, r0, #0x10
    add     r0, r0, r1
    mov     r3, r2

    sub     r4, r4, #2
    cmp     r4, #1
    bgt     loop_h2
loop_h1:

    add     r6, r0, r1
    add     r7, r0, r1, lsl #0x1
    add     r8, r6, r1, lsl #0x1
    add     r9, r0, r1, lsl #0x2
    lsr     r11, r5, #0x2
    cmp     r11, #0
    beq     loop_h1_w1

loop_h1_w4:
    vldm        r0, {d24-d27}
    add         r0, r0, #0x10
    vext.32     q9, q12, q13, #1
    vmul.f32    q10, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d1[0]
    vmla.f32    q10, q13, d2[0]
    vldm        r6, {d24-d27}
    vmla.f32    q10, q9, d1[1]

    add         r6, r6, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d2[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d3[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d3[1]
    vmla.f32    q10, q13, d4[1]
    vldm        r7, {d24-d27}
    vmla.f32    q10, q9, d4[0]

    add         r7, r7, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d5[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d5[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d6[0]
    vmla.f32    q10, q13, d7[0]
    vldm        r8, {d24-d27}
    vmla.f32    q10, q9, d6[1]

    add         r8, r8, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d7[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d8[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d8[1]
    vmla.f32    q10, q13, d9[1]
    vldm        r9, {d24-d27}
    vmla.f32    q10, q9, d9[0]

    add         r9, r9, #0x10
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d10[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d10[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d11[0]
    vmla.f32    q10, q9, d11[1]
    vmla.f32    q10, q13, d12[0]

    vadd.f32    q10, q10, q15
    cmp         r12, #0

    blt         loop_h1_w4_save
	vmax.f32	q10, q10, q14
    beq         loop_h1_w4_save
	vmin.f32	q10, q10, q7

loop_h1_w4_save:
    vstm    r3!, {d20-d21}
    subs    r11, r11, #0x1
    bne     loop_h1_w4

loop_h1_w1:
    and     r11, r5, #0x3
    cmp     r11, #0
    beq     loop_h1_end

    vldm        r0, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmul.f32    q10, q12, d0[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d0[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d1[0]
    vmla.f32    q10, q9, d1[1]
    vmla.f32    q10, q13, d2[0]

    vldm        r6, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d2[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d3[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d3[1]
    vmla.f32    q10, q9, d4[0]
    vmla.f32    q10, q13, d4[1]

    vldm        r7, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d5[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d5[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d6[0]
    vmla.f32    q10, q9, d6[1]
    vmla.f32    q10, q13, d7[0]

    vldm        r8, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d7[1]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d8[0]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d8[1]
    vmla.f32    q10, q9, d9[0]
    vmla.f32    q10, q13, d9[1]

    vldm        r9, {d24-d27}
    vext.32     q9, q12, q13, #1
    vmla.f32    q10, q12, d10[0]
    vext.32     q8, q12, q13, #2
    vmla.f32    q10, q9, d10[1]
    vext.32     q9, q12, q13, #3
    vmla.f32    q10, q8, d11[0]
    vmla.f32    q10, q9, d11[1]
    vmla.f32    q10, q13, d12[0]

    vadd.f32    q10, q10, q15
    cmp         r12, #0

    blt         loop_h1_w1_save
	vmax.f32	q10, q10, q14
    beq         loop_h1_w1_save
	vmin.f32	q10, q10, q7
loop_h1_w1_save:
    vmov.f32    d13, d20
    vext.32     q10, q10, q10, #1
    vstr        s26, [r3]
    add         r3, r3, #0x4
    subs        r11, r11, #0x1
    bne         loop_h1_w1_save

loop_h1_end:
	vpop		{d8 - d15}
	pop		{r4 - r12, pc}
    .end
