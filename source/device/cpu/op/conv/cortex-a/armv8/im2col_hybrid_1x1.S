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
//         x0 arg0  input address 
//         x1 arg1  input_xy
//         x2 arg2  col address
//         x3 arg3  col_cnt must be multiply of 4
//         x4 arg4  kernel_size
//         x5 arg5  scale address
//
// register definition
//    x0 input address 
//    x1 input_xy x 4
//    x2 col address
//    x3 col_cnt
//    x4 kernel_size
//    x5 scale address
//    x6 input start pointer
//    x7 input pointer
//    x9 channel cnt
//    x11 input address + input_xy
//    x12 col + 8
//    x13 input address + input_xy * 2
//    x14 input address + input_xy * 3

        .section .text,"ax"
        .align 5

        .type   im2col_hybrid_1x1 STT_FUNC
        .global im2col_hybrid_1x1
        .hidden im2col_hybrid_1x1
im2col_hybrid_1x1:
	cmp	x3, 4
	b.lt	col_end
	lsr	x3, x3, 2	// x3 = col_cnt
	ldr	s31,[x5]
	mov	x6, x0
	lsl	x1, x1, 2	// x1 = input_xy size
	// col loop
col_loop:
	mov	x7, x6		// x7 = input
	lsr	x9, x4, 2	// x9 = channel cnt / 4
	add	x11,x7, x1
	add	x13,x7, x1, LSL 1
	add	x12, x2, 8
	and	x10, x4, 0x2
	add	x14,x13,x1
	cbz	x9, kernel_loop4_end
	// kernel size loop
kernel_loop4:
	ldr	q0, [x7]
	ldr	q1, [x11]
	ldr	q4, [x13]
	ldr	q5, [x14]
	subs	x9, x9, 1
	fmul	v0.4s, v0.4s, v31.s[0]
	prfm	pldl1keep, [x7, 0x40]
	fmul	v1.4s, v1.4s, v31.s[0]
	fmul	v4.4s, v4.4s, v31.s[0]
	prfm	pldl1keep, [x11,0x40]
	fmul	v5.4s, v5.4s, v31.s[0]
	fcvtas	v0.4s, v0.4s
	prfm	pldl1keep, [x13,0x40]
	fcvtas	v1.4s, v1.4s
	prfm	pldl1keep, [x14,0x40]
	fcvtas	v4.4s, v4.4s
	add	x7, x7, x1, LSL 2
	fcvtas	v5.4s, v5.4s
	zip1	v2.16b, v0.16b, v1.16b
	add	x11,x11, x1, LSL 2
	zip2	v3.16b, v0.16b, v1.16b
	add	x13,x13, x1, LSL 2
	zip1	v6.16b, v4.16b, v5.16b
	add	x14,x14, x1, LSL 2
	zip2	v7.16b, v4.16b, v5.16b
	st1	{v2.h}[0], [x2], #2
	st1	{v6.h}[0], [x12],#2
	st1	{v2.h}[4], [x2], #2
	st1	{v6.h}[4], [x12],#2
	st1	{v3.h}[0], [x2], #2
	st1	{v7.h}[0], [x12],#2
	st1	{v3.h}[4], [x2]
	add	x2, x2, 10
	st1	{v7.h}[4], [x12]
	add	x12,x12,10
	b.ne	kernel_loop4

kernel_loop4_end:
	and	x13, x4, 0x1
	cbz	x10, kernel_last1

	ldr	q0, [x7]
	ldr	q1, [x11]
	add	x7, x7, x1, LSL 1
	fmul	v0.4s, v0.4s, v31.s[0]
	prfm	pldl1keep, [x7, 0x40]
	fmul	v1.4s, v1.4s, v31.s[0]
	fcvtas	v0.4s, v0.4s
	prfm	pldl1keep, [x11,0x40]
	fcvtas	v1.4s, v1.4s
	zip1	v2.16b, v0.16b, v1.16b
	zip2	v3.16b, v0.16b, v1.16b
	st1	{v2.h}[0], [x2], #2
	st1	{v2.h}[4], [x2], #2
	st1	{v3.h}[0], [x2], #2
	st1	{v3.h}[4], [x2], #2

kernel_last1:
	cbz	x13, kernel_loop_end

	// last channel
	ldr	q0, [x7]
	movi	d1, 0	
	fmul	v0.4s, v0.4s, v31.s[0]
	prfm	pldl1keep, [x7, 0x40]
	fcvtas	v0.4s, v0.4s
	zip1	v2.16b, v0.16b, v1.16b
	zip2	v3.16b, v0.16b, v1.16b
	st1	{v2.h}[0], [x2], #2
	st1	{v2.h}[4], [x2], #2
	st1	{v3.h}[0], [x2], #2
	st1	{v3.h}[4], [x2], #2

kernel_loop_end:
	add	x6, x6, 16
	subs	x3, x3, 1
	b.ne	col_loop

col_end:

	ret
	.end
