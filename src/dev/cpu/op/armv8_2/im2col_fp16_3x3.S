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

        .type   im2col_fp16_3x3 STT_FUNC
        .global im2col_fp16_3x3
        .hidden im2col_fp16_3x3
im2col_fp16_3x3:
	// initial
	cbz	x3, finish
	cmp	x5, 2
	lsl	x1, x1, 1	// x1 = input_x size
	mul	x2, x2, x1	// x2 = input_xy size
	add	x11,x0, x1
	add	x12,x0, x1, LSL 1
	beq	stride2_channel_loop

stride1_channel_loop:
	ldr	d0,  [x0]	
	ldr	s1,  [x0, 0x8]	
	ldr	d2,  [x11]	
	ldr	s3,  [x11,0x8]	
	ldr	d4,  [x12]	
	ldr	s5,  [x12,0x8]	
	subs	x3, x3, 1
	ext	v16.8b, v0.8b, v1.8b, 2
	prfm	pldl1strm, [x0, 0x40]
	ext	v17.8b, v0.8b, v1.8b, 4
	ext	v19.8b, v2.8b, v3.8b, 2
	prfm	pldl1strm, [x11,0x40]
	ext	v20.8b, v2.8b, v3.8b, 4
	ext	v22.8b, v4.8b, v5.8b, 2
	prfm	pldl1strm, [x12,0x40]
	ext	v23.8b, v4.8b, v5.8b, 4
	stp	d0, d16, [x4], 0x10
	add	x0, x0, x2
	stp	d17,d2,  [x4], 0x10
	add	x11,x11,x2
	stp	d19,d20, [x4], 0x10
	add	x12,x12,x2
	stp	d4, d22, [x4], 0x10
	str	d23, [x4], 0x8
	bne	stride1_channel_loop
	b	finish

stride2_channel_loop:
	ld2	{v16.4h,v17.4h}, [x0]	
	ldr	h1,  [x0, 0x10]	
	ld2	{v19.4h,v20.4h}, [x11]	
	ldr	h3,  [x11,0x10]	
	ld2	{v22.4h,v23.4h}, [x12]	
	ldr	h5,  [x12,0x10]	
	subs	x3, x3, 1
	prfm	pldl1strm, [x0, 0x40]
	ext	v18.8b,v16.8b, v1.8b, 2
	prfm	pldl1strm, [x11,0x40]
	ext	v21.8b,v19.8b, v3.8b, 2
	prfm	pldl1strm, [x12,0x40]
	ext	v24.8b,v22.8b, v5.8b, 2
	stp	d16, d17, [x4], 0x10
	add	x0, x0, x2
	stp	d18, d19, [x4], 0x10
	add	x11,x11,x2
	stp	d20, d21, [x4], 0x10
	add	x12,x12,x2
	stp	d22, d23, [x4], 0x10
	str	d24, [x4], 0x8
	bne	stride2_channel_loop
finish:
	ret

	.end
