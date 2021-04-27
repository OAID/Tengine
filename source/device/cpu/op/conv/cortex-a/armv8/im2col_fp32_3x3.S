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
 * Author: xiaowei@openailab.com
 */
//
// im2col fp16 for kernel 3x3  include 2 function  stride 1 and stride 2
// ABCDABCD
//
// input:
//         x0 arg0  input address 
//         x1 arg1  input_x
//         x2 arg2  input_y
//         x3 arg3  input channel cnt
//         x4 arg4  col address
//         x5 arg5  stride_x
//
// register definition
//    x0 cl0 address  q0  q1    d16 d17 d18
//    x1 input_x x 4
//    x2 input_xy x 4
//    x3 input channel
//    x4 col address
//    x5 stride_x
//    x11 cl1 address q2  q3    d19 d20 d21
//    x12 cl2 address q4  q5    d22 d23 d24

        .section .text,"ax"
        .align 5

        .type   im2col_fp32_3x3 STT_FUNC
        .global im2col_fp32_3x3
        .hidden im2col_fp32_3x3
im2col_fp32_3x3:
	// initial
	cbz	x3, finish
	cmp	x5, 2
	lsl	x1, x1, 2	// x1 = input_x size
	mul	x2, x2, x1	// x2 = input_xy size
	add	x11,x0, x1
	add	x12,x0, x1, LSL 1
	beq	stride2_channel_loop

stride1_channel_loop:
	ldr	q0,  [x0]	
	ldr	d1,  [x0, 0x10]	
	ldr	q2,  [x11]	
	ldr	d3,  [x11,0x10]	
	ldr	q4,  [x12]	
	ldr	d5,  [x12,0x10]	
	subs	x3, x3, 1
	ext	v16.16b, v0.16b, v1.16b, 4
	prfm	pldl1strm, [x0, 0x40]
	ext	v17.16b, v0.16b, v1.16b, 8
	add	x0, x0, x2
	ext	v19.16b, v2.16b, v3.16b, 4
	prfm	pldl1strm, [x11,0x40]
	ext	v20.16b, v2.16b, v3.16b, 8
	add	x11,x11,x2
	ext	v22.16b, v4.16b, v5.16b, 4
	prfm	pldl1strm, [x12,0x40]
	ext	v23.16b, v4.16b, v5.16b, 8
	add	x12,x12,x2
	stp	q0, q16, [x4], 0x20
	stp	q17,q2,  [x4], 0x20
	stp	q19,q20, [x4], 0x20
	stp	q4, q22, [x4], 0x20
	str	q23, [x4], 0x10
	bne	stride1_channel_loop
	b	finish

stride2_channel_loop:
	ld2	{v16.4s, v17.4s}, [x0]
	ldr	s18, [x0, 0x20]	
	ld2	{v19.4s, v20.4s}, [x11]
	ldr	s21, [x11,0x20]	
	ld2	{v22.4s, v23.4s}, [x12]
	ldr	s24, [x12,0x20]	
	subs	x3, x3, 1
	ext	v18.16b,v16.16b, v18.16b, 4
	prfm	pldl1strm, [x0, 0x60]
	ext	v21.16b,v19.16b, v21.16b, 4
	prfm	pldl1strm, [x11,0x60]
	ext	v24.16b,v22.16b, v24.16b, 4
	prfm	pldl1strm, [x12,0x60]
	stp	q16, q17, [x4], 0x20
	add	x0, x0, x2
	stp	q18, q19, [x4], 0x20
	add	x11,x11,x2
	stp	q20, q21, [x4], 0x20
	add	x12,x12,x2
	stp	q22, q23, [x4], 0x20
	str	q24, [x4], 0x10
	bne	stride2_channel_loop
finish:
	ret

	.end
