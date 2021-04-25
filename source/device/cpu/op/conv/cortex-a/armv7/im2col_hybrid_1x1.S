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
 * Copyright (c) 2018, Open AI Lab
 * Author: xiaowei@openailab.com
 */
//
// im2col for kernel 1x1 s1p0d1
//
// input:
//         r0     arg0  input address 
//         r1     arg1  input_xy
//         r2     arg2  col address
//         r3     arg3  col_cnt must be multiply of 4
//         sp     arg4  input channel
//         sp+0x4 arg5  scale address
//
// register definition
//    r0 input address 
//    r1 input_xy x 4
//    r2 col address
//    r3 col_cnt
//    r4 input pointer
//    r5 channel cnt
//    r6 input address + input_xy
//
//    q0-1 input
//    s8   scale

        .section .text,"ax"
        .align 5

        .type   im2col_hybrid_1x1 STT_FUNC
        .global im2col_hybrid_1x1
        .hidden im2col_hybrid_1x1
im2col_hybrid_1x1:
	push		{r4-r6,lr}
	cmp		r3, #4
	blt		col_end
	lsr		r3, r3, #2	// r3 = col_cnt

	ldr		r4, [sp, #0x14]
	ldr		r5, [sp, #0x10]
	lsl		r1, r1, #2	// r1 = input_xy size
	vldr		s8,[r4]
	// col loop
col_loop:
	mov		r4, r0		// r4 = input
	lsr		r5, r5, #1	// r5 = channel cnt / 2
	add		r6, r4, r1	// r6 = input + input_xy
	cmp		r5, #0
	beq		last_channel

	// kernel size loop
channel_loop2:
	vldr		d0, [r4]
	vldr		d2, [r6]
	vldr		d1, [r4, #0x8]
	vldr		d3, [r6, #0x8]
	subs		r5, r5, #1
	vmul.f32	d0, d0, d4[0]
	vmul.f32	d2, d2, d4[0]
	vmul.f32	d1, d1, d4[0]
	vmul.f32	d3, d3, d4[0]
	vcvtr.s32.f32	s0, s0
	vcvtr.s32.f32	s1, s1
	vcvtr.s32.f32	s4, s4
	vcvtr.s32.f32	s5, s5
	vcvtr.s32.f32	s2, s2
	vcvtr.s32.f32	s3, s3
	vcvtr.s32.f32	s6, s6
	vcvtr.s32.f32	s7, s7
	vzip.8		d0, d2
	vzip.8		d1, d3
	pld		[r4, #0x40]
	add		r4, r4, r1, LSL #1
	pld		[r6, #0x40]
	add		r6, r6, r1, LSL #1
	vst2.16		{d0[0],d2[0]},[r2]!
	vst2.16		{d1[0],d3[0]},[r2]!
	bne		channel_loop2

last_channel:
	ldr		r5, [sp, #0x10]
	tst		r5, #1
	beq		channel_end

	// last channel
	vldr		d0, [r4]
	vmov.i64	d2, #0	
	vldr		d1, [r4, #0x8]
	vmov.i64	d3, #0	
	vmul.f32	q0, q0, d4[0]
	vcvtr.s32.f32	s0, s0
	vcvtr.s32.f32	s1, s1
	vcvtr.s32.f32	s2, s2
	vcvtr.s32.f32	s3, s3
	vzip.8		d0, d2
	vzip.8		d1, d3
	pld		[r4, #0x40]
	vst2.16		{d0[0],d2[0]},[r2]!
	vst2.16		{d1[0],d3[0]},[r2]!

channel_end:
	add		r0, r0, #16
	subs		r3, r3, #1
	bne		col_loop

col_end:
	pop		{r4-r6,pc}

	.end
