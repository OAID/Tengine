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
 * Author: haoluo@openailab.com
 */
 
// input:
//         x0     arg0  biases address {b0,b1,b2,b3}   nullptr means no biases
//         x1     arg1  input  address 
//         x2     arg2  kernel address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],k[0-15][4],...}
//         x3     arg4  output address
//                        direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                     output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//										....
//         x4     arg3  input_chan
//         x5 	  arg5  output x
//		   x6 	  arg6	output y
//         x7	  arg7  activation flag   relu layers is integrated after convolution
//
// output: no
//
//	x8	loop_chan
//	x9  loop_w
//	x10 tmp
//  x11 input_cur
//  x12 kernel_cur
//  x13 output_cur
//  x14 h*w*4
//  x15 loop_w_less
//	x16 loop_h
//
//
// v0   4S data of input0   {i3   i2   i1   i0} 
// v1   4S data of input1   {i7   i6   i5   i4}
// v4   4S kernal data      {k3 | k2 | k1 | k0}
// v5   4S kernal data      {k7 | k6 | k5 | k4}
// v6   4S kernal data      {kb | ka | k9 | k8}
// v7   4S kernal data      {kf | ke | kd | kc}
//
// v16 dot product for {i0k3  i0k2  i0k1  i0k0}
// v17 dot product for {i1k3  i1k2  i1k1  i1k0}
// v18 dot product for {i2k3  i2k2  i2k1  i2k0}
// v19 dot product for {i3k3  i3k2  i3k1  i3k0}
//
// v20 dot product for {i0k7  i0k6  i0k5  i0k4}
// v21 dot product for {i1k7  i1k6  i1k5  i1k4}
// v22 dot product for {i2k7  i2k6  i2k5  i2k4}
// v23 dot product for {i3k7  i3k6  i3k5  i3k4}
//
// v24 dot product for {i0kb  i0ka  i0k9  i0k8}
// v25 dot product for {i1kb  i1ka  i1k9  i1k8}
// v26 dot product for {i2kb  i2ka  i2k9  i2k8}
// v27 dot product for {i3kb  i3ka  i3k9  i3k8}
//
// v28 dot product for {i0kf  i0ke  i0kd  i0kc}
// v29 dot product for {i1kf  i1ke  i1kd  i1kc}
// v30 dot product for {i2kf  i2ke  i2kd  i2kc}
// v31 dot product for {i3kf  i3ke  i3kd  i3kc}


	.section .text, "ax"
	.align 5

	.type direct_k3s1p1_4x16_a72 STT_FUNC
	.global direct_k3s1p1_4x16_a72
	.hidden direct_k3s1p1_4x16_a72

direct_k3s1p1_4x16_a72:

	
//---------------------------------------------------   first  row ---------------------------------------
row_first_1:
	// biases_initial
	prfm	pldl1keep, [x2,0xc0]
	cbz	x0, none_biases1
		ldr	q16, [x0]
		ldr	q20, [x0, 0x10]
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q24, [x0, 0x20]
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q28, [x0, 0x30]
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
	b	convolution_start1
	
none_biases1:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start1:
	add  x11, x1,  0x0
	add  x12, x2,  0x0
	mul  x14, x5, x6
	lsl  x14, x14, 0x2
	
	mov  x8, x4
	
loop_channel_1:
	ldr  q0, [x11] 
	ldr	 q4, [x12,0xc0]
	ldr  q5, [x12,0xd0]		// k3[7-0]
	subs x8, x8, 0x1
	fmla	v17.4s, v4.4s,  v0.s[0]		// k3[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k3[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k3[3-0]i[3]
	ldr  q1, [x11, 0x10]
	ldp	 q6, q7, [x12,0xe0]		// k3[f-8]
	fmla	v21.4s, v5.4s,  v0.s[0]		// k3[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k3[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k3[7-4]i[3]
	LSL   x10, x14, 0x2
	ldp	 q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k3[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k3[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k3[b-8]i[3]
	prfm	pldl1keep, [x12, 0x300]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k3[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k3[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k3[f-c]i[3]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	prfm	pldl1keep, [x11, x10]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	add   x10, x10, x5, LSL 0x2
	ldp	 q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k4[f-c]i[3]
	ldp	 q6, q7, [x12,0x160]		// k5[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k5[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	prfm	pldl1keep, [x11, x10]
	ldp	 q4, q5, [x12,0x180]		// k6[7-0]
	lsl	 x10, x5 ,0x2
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	ldr	 q0, [x11,x10]
	add  x10, x10 ,0x10
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k5[f-c]i[3]
	ldp  q6, q7, [x12,0x1a0]		// k6[f-8]
	
	fmla	v17.4s, v4.4s,  v0.s[0]		//k6[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		//k6[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k[7-4]i[3]
	ldr	 q1, [x11,x10]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k6[f-c]i[3]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k7[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k7[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x200]		// k8[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k7[f-c]i[3]
	ldp	q6, q7, [x12,0x220]		// k8[f-8]
	
	lsl  x10, x14, 0x3
	fmla	v16.4s, v4.4s,  v0.s[1]		// k8[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	add  x10,  x10,  x5, lsl 0x3
	fmla	v20.4s, v5.4s,  v0.s[1]		// k8[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	prfm	pldl1keep, [x11, x10]
	
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k8[f-c]i[3]
	
	add x11, x11, x14
	add x12, x12, 0x240
	
	b.ne  loop_channel_1
	
activation_1:
	add x13, x3 , 0x0
	cmp w7, 0
	blt	save_result_1

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s

	beq   save_result_1

	dup     v1.4s, v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s

	
save_result_1:

	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x13], x14
	
	add  x13, x3, 0x10

	add  x11, x1, 0x0c
	add x12, x2, 0x0		// kernel
	prfm	pldl1keep, [x12]
	
row_loop4_2:
	
	sub  x15, x5, 0x5		//input_w - 4 - 1 
	lsr  x9, x15, 0x2
	cbz  x9, loop_col_less_2
loop_col_2:
	// biases_initial
	cbz	x0, none_biases2
		ldr	q16, [x0], 0x10
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q20, [x0], 0x10
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q24, [x0], 0x10
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		ldr	q28, [x0], 0x10
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
		sub  x0, x0, 0x40
	b	convolution_start2
	
none_biases2:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start2:
	add x8, x4, 0x0		// input_chan

loop_channel_2:
	ldr q0, [x11]
	ldr q1, [x11,0x10]
	
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	prfm	pldl1keep, [x12, 0x480]
	prfm	pldl1keep, [x11, x14]
	subs x8, x8, 0x1
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k3[f-c]i[3]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k4[f-c]i[3]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	lsl	x10, x5 ,0x2
	fmla	v16.4s, v4.4s,  v0.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	ldr	q0, [x11,x10]
	add x10, x10 ,0x10
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k5[f-c]i[3]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	ldr	q1, [x11,x10]
	add x11, x11, x14
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k6[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k6[f-c]i[3]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k7[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x200]		// k8[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k7[f-c]i[3]
	ldp	q6, q7, [x12,0x220]		// k8[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k8[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k8[f-c]i[3]
	add x12, x12, 0x240
	b.ne  loop_channel_2
	
activation_2:

    cmp  w7,0
	blt save_result_2

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s

	beq  save_result_2
    dup v1.4s, v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s

	
save_result_2:
	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x10
	
loop_col_end_2:
	mul  x10, x14, x4 
	add  x12, x2, 0x0
	sub  x11, x11, x10
	prfm	pldl1keep, [x12, 0xc0]
	add  x11, x11, 0x10
	
	prfm	pldl1keep, [x11]
	subs x9, x9, 0x1
	b.ne  loop_col_2
	
loop_col_less_2:

	and x9, x15, 0x3
	cbz x9, row_end_3
	
loop_col_less_start_2:
	add x8, x4, 0x0
	cbz	x0, none_biases2_1
		ldr	q16, [x0]
		ldr	q17, [x0,0x10]
		ldr	q18, [x0,0x20]
		ldr	q19, [x0,0x30]
	b	loop_channel_2_1
	
none_biases2_1:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
loop_channel_2_1:
	ldr q0, [x11]
	lsl	x10, x5 ,0x2
	ldr q1, [x11, x10]
	ldr q4, [x12, 0xc0]
	ldr q5, [x12, 0xd0]		// k3[7-0]
	ldp q6, q7, [x12, 0xe0]		// k3[f-8]
	subs x8, x8, 0x1
	prfm	pldl1keep, [x12, 0x300]
	prfm	pldl1keep, [x11, x14]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v18.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[0]		// k3[f-c]i[0]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	fmla	v16.4s, v4.4s,  v0.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v18.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[1]		// k4[f-c]i[0]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	fmla	v16.4s, v4.4s,  v0.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v18.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[2]		// k5[f-c]i[0]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	
	fmla	v16.4s, v4.4s,  v1.s[0]		// k6[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v1.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[0]		// k6[f-c]i[0]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	fmla	v16.4s, v4.4s,  v1.s[1]		// k7[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x200]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v1.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[1]		// k7[f-c]i[0]
	ldp	q6, q7, [x12,0x220]		// k7[f-8]
	fmla	v16.4s, v4.4s,  v1.s[2]		// k8[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[2]		// k[7-4]i[0]
	fmla	v18.4s, v6.4s,  v1.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[2]		// k8[f-c]i[0]
	add x11, x11, x14
	add x12, x12, 0x240
	
	b.ne  loop_channel_2_1
	
activation_2_1:
	cmp w7,0
	blt	save_result_2_1

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

	beq save_result_2_1

	dup     v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	
save_result_2_1:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x4
	
loop_col_less_end_2:
	mul  x10, x14, x4 
	add  x12, x2, 0x0
	sub  x11, x11, x10
	prfm	pldl1keep, [x12,0xc0]
	add  x11, x11, 0x4
	prfm	pldl1keep, [x11]
	
	subs x9, x9, 0x1
	b.ne  loop_col_less_start_2
	

row_end_3:
	cbz	x0, none_biases3
		ldr	q16, [x0]
		ldr	q17, [x0,0x10]
		ldr	q18, [x0,0x20]
		ldr	q19, [x0,0x30]
	b	convolution_start3
	
none_biases3:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
convolution_start3:
	add  x8,  x4, 0x0
	
loop_channel_3:
	ldr q0, [x11] 
	lsl	x10, x5 ,0x2
	ldr q1,	[x11, x10]
	ldr	q4, [x12, 0xc0]
	ldr q5, [x12,0xd0]		// k3[7-0]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	prfm	pldl1keep, [x12, 0x300]
	prfm	pldl1keep, [x11, x14]
	fmla	v16.4s, v4.4s,  v0.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v18.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[0]		// k3[f-c]i[0]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	fmla	v16.4s, v4.4s,  v0.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v18.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[1]		// k4[f-c]i[0]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	
	
	fmla	v16.4s, v4.4s,  v1.s[0]		// k6[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v1.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[0]		// k6[f-c]i[0]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	fmla	v16.4s, v4.4s,  v1.s[1]		// k7[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[1]		// k[7-4]i[0]
	subs x8, x8, 0x1
	fmla	v18.4s, v6.4s,  v1.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[1]		// k7[f-c]i[0]
	add x11, x11, x14
	add x12, x12, 0x240
	
	b.ne  loop_channel_3

activation_3:
    cmp w7, 0
	blt save_result_3

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

    beq    save_result_3

	dup     v1.4s,v1.s[0]
	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s

	
save_result_3:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x4
	
	add x11, x1, 0x0
	add x12, x2, 0x0
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]

//------------------------------------------------- middle row --------------------------------------------------------
	
	sub x6, x6, 0x2
loop_middle_row:

row_first_4:
	// biases_initial
	cbz	x0, none_biases4
		ldr	q16, [x0], 0x10
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q20, [x0], 0x10
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q24, [x0], 0x10
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		ldr	q28, [x0], 0x10
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
		sub  x0, x0, 0x40
	b	convolution_start4
	
none_biases4:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start4:
	add  x8,  x4, 0x0
	
loop_channel_4:
	ldr q0, [x11] 
	ldr q1,	[x11, 0x10]
	ldr	q4, [x12 ]
	ldr q5, [x12, 0x10]		// k0[7-0]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	prfm	pldl1keep, [x12, 0x480]
	fmla	v17.4s, v4.4s,  v0.s[0]		// k0[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k0[f-c]i[3]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k1[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x80]		// k2[7-0]
	lsl x10, x14, 0x3
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k1[f-c]i[3]
	ldp	q6, q7, [x12,0xa0]		// k2[f-8]
	prfm	pldl1keep, [x11, x10]
	add  x10, x10, x5, lsl 0x2
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k2[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k5[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	prfm	pldl1keep, [x11, x10]
	
	lsl	x10, x5 ,0x2
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	ldr	q0, [x11,x10]
	add x10, x10 ,0x10
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k2[f-c]i[3]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	ldr	q1, [x11,x10]
	
	fmla	v17.4s, v4.4s,  v0.s[0]		// k3[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k3[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k3[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		// k3[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k3[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k3[7-4]i[3]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k3[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k3[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k3[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k3[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k3[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k3[f-c]i[3]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k4[f-c]i[3]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k5[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	lsl	 x10, x5 ,0x3
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	ldr	 q0, [x11,x10]
	add  x10, x10 ,0x10
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k5[f-c]i[3]
	ldp	 q6, q7, [x12,0x1a0]		// k6[f-8]
	
	fmla	v17.4s, v4.4s,  v0.s[0]		//k6[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		//k6[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k[7-4]i[3]
	ldr	 q1, [x11,x10]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k6[f-c]i[3]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k7[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k7[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x200]		// k8[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k7[f-c]i[3]
	ldp	q6, q7, [x12,0x220]		// k8[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k8[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k8[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k8[f-c]i[3]
	
	add x11, x11, x14
	add x12, x12, 0x240
	
	subs  x8, x8, 0x1
	b.ne  loop_channel_4
	
activation_4:
    cmp  w7, 0
	blt	 save_result_4

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s

	beq save_result_4

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s
	
save_result_4:
	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x10
	
	mul  x10, x14, x4
	sub  x11, x11, x10
	add  x11, x11, 0xc
	add  x12, x2, 0x0		// kernel
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	
row_loop4_5:
	
	lsr  x9, x15, 0x2
	cbz  x9, loop_col_less_5
	
loop_col_5:
	// biases_initial
	cbz	x0, none_biases5
		ldr	q16, [x0]
		ldr	q20, [x0, 0x10]
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q24, [x0, 0x20]
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q28, [x0, 0x30]
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
	b	convolution_start5
	
none_biases5:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start5:
	mov  x8, x4   		// input_chan

loop_channel_5:
	ldr q0, [x11]
	ldp	q4, q5, [x12,0x00]		// k0[7-0]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k0[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x240]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	ldr q1, [x11,0x10]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	prfm	pldl1keep, [x11, x14]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k0[f-c]i[3]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k1[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x280]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x80]		// k2[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	prfm	pldl1keep, [x12, 0x2c0]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k1[f-c]i[3]
	ldp	q6, q7, [x12,0xa0]		// k2[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k2[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x300]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	lsl	x10, x5, 0x2
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	ldr	q0, [x11, x10]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	add x10, x10, 0x10
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k2[f-c]i[3]
	ldr	q1, [x11, x10]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	lsl x10, x14, 0x1
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	prfm	pldl1keep, [x11, x10]
	add x10, x10, x14
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	prfm	pldl1keep, [x11, x10]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k3[f-c]i[3]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x340]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	prfm	pldl1keep, [x12, 0x380]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k4[f-c]i[3]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x3c0]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	lsl	x10, x5 ,0x3
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	ldr	q0, [x11, x10]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	add x10, x10 ,0x10
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k5[f-c]i[3]
	ldr	q1, [x11, x10]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k6[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	prfm	pldl1keep, [x12, 0x400]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	prfm	pldl1keep, [x12, 0x440]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k6[f-c]i[3]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k7[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x200]		// k8[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k7[f-c]i[3]
	ldp	q6, q7, [x12,0x220]		// k8[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k8[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	add x12, x12, 0x240
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	add x11, x11, x14
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k8[f-c]i[3]
	subs  x8, x8, 0x1
	
	b.ne  loop_channel_5
	
activation_5:
    cmp      w7,0
	blt	save_result_5

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s
	
	beq save_result_5

	dup   v1.4s,v1.s[0]
	
	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s
	
save_result_5:
	add x16, x13, x14, lsl 0x2
	add x17, x13, x14, LSL 0x3
	add x18, x16, x14, LSL 0x3
	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x16], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x17], x14  
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x18], x14  
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x16], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x17], x14 
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x18], x14 
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x16], x14  
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x17], x14  
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x18], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x16]  
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x17] 
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x18]
	sub  x13, x13, x14, LSL 0x2
	add  x13, x13, 0x10
	
loop_col_end_5:
	mul  x10, x14, x4 
	add  x12, x2, 0x0
	sub  x11, x11, x10
	subs x9, x9, 0x1
	add  x11, x11, 0x10

	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	b.ne  loop_col_5
	
loop_col_less_5:
	and  x9, x15, 0x3
	cbz  x9, row_end_6
	
loop_col_less_start_5:
	add x8, x4, 0x0
	cbz	x0, none_biases5_1
	ldr	q16, [x0]
	ldr	q17, [x0,0x10]
	ldr	q18, [x0,0x20]
	ldr	q19, [x0,0x30]
	b	loop_channel_5_1
	
none_biases5_1:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
loop_channel_5_1:
	ldr q0, [x11]
	lsl	x10, x5 ,0x2
	ldr q1, [x11, x10]
	lsl	x10, x5 ,0x3
	ldr q2, [x11, x10]
	ldp q4, q5, [x12, 0x00]		// k0[7-0]
	ldp q6, q7, [x12, 0x20]		// k0[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k0[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v18.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[0]		// k0[f-c]i[0]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	fmla	v16.4s, v4.4s,  v0.s[1]		// k1[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x80]		// k2[7-0]
	fmla	v18.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[1]		// k1[f-c]i[0]
	ldp	q6, q7, [x12,0xa0]		// k2[f-8]
	fmla	v16.4s, v4.4s,  v0.s[2]		// k2[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v18.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[2]		// k2[f-c]i[0]
	ldp q6, q7, [x12,0xe0]		// k3[f-8]
	fmla	v16.4s, v4.4s,  v1.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v18.4s, v6.4s,  v1.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[0]		// k3[f-c]i[0]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	fmla	v16.4s, v4.4s,  v1.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v18.4s, v6.4s,  v1.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[1]		// k4[f-c]i[0]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	fmla	v16.4s, v4.4s,  v1.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v18.4s, v6.4s,  v1.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[2]		// k5[f-c]i[0]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	fmla	v16.4s, v4.4s,  v2.s[0]		// k6[3-0]i[0]
	fmla	v17.4s, v5.4s,  v2.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v2.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v2.s[0]		// k6[f-c]i[0]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	fmla	v16.4s, v4.4s,  v2.s[1]		// k7[3-0]i[0]
	fmla	v17.4s, v5.4s,  v2.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x200]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v2.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v2.s[1]		// k7[f-c]i[0]
	ldp	q6, q7, [x12,0x220]		// k7[f-8]
	fmla	v16.4s, v4.4s,  v2.s[2]		// k8[3-0]i[0]
	fmla	v17.4s, v5.4s,  v2.s[2]		// k[7-4]i[0]
	fmla	v18.4s, v6.4s,  v2.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v2.s[2]		// k8[f-c]i[0]
	add  x11, x11, x14
	add  x12, x12, 0x240
	
	subs x8, x8, 0x1
	
	b.ne  loop_channel_5_1
	
activation_5_1:
    cmp w7,0
	blt	save_result_5_1

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

	beq save_result_5_1

	dup     v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s

	
save_result_5_1:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x4
	
loop_col_less_end_5:
	mul  x10, x14, x4 
	sub  x11, x11, x10
	add  x11, x11, 0x4
	add  x12, x2,  0x0
	
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	
	subs x9, x9, 0x1
	b.ne  loop_col_less_start_5
	
row_end_6:
	cbz	x0, none_biases6
		ldr	q16, [x0]
		ldr	q17, [x0,0x10]
		ldr	q18, [x0,0x20]
		ldr	q19, [x0,0x30]
	b	convolution_start6
	
none_biases6:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
convolution_start6:
	sub  x11, x11, 0x8
	add  x8,  x4, 0x0
	
loop_channel_6:
	ldr q0, [x11] 
	lsl x10, x5, 0x2
	ldr q1,	[x11, x10]
	lsl x10, x5, 0x3
	ldr q2, [x11, x10]
	ldp	q4, q5, [x12,0x00]		// k0[7-0]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	fmla	v16.4s, v4.4s,  v0.s[2]		// k0[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v18.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[2]		// k0[f-c]i[0]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[3]		// k1[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[3]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v18.4s, v6.4s,  v0.s[3]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[3]		// k1[f-c]i[0]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	
	fmla	v16.4s, v4.4s,  v1.s[2]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v18.4s, v6.4s,  v1.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[2]		// k3[f-c]i[0]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	
	fmla	v16.4s, v4.4s,  v1.s[3]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[3]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x180]		// k6[7-0]
	fmla	v18.4s, v6.4s,  v1.s[3]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[3]		// k4[f-c]i[0]
	ldp	q6, q7, [x12,0x1a0]		// k6[f-8]
	
	fmla	v16.4s, v4.4s,  v2.s[2]		// k6[3-0]i[0]
	fmla	v17.4s, v5.4s,  v2.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x1c0]		// k7[7-0]
	fmla	v18.4s, v6.4s,  v2.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v2.s[2]		// k6[f-c]i[0]
	ldp	q6, q7, [x12,0x1e0]		// k7[f-8]
	
	fmla	v16.4s, v4.4s,  v2.s[3]		// k7[3-0]i[0]
	fmla	v17.4s, v5.4s,  v2.s[3]		// k[7-4]i[0]
	fmla	v18.4s, v6.4s,  v2.s[3]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v2.s[3]		// k7[f-c]i[0]
	
	add x11, x11, x14
	add x12, x12, 0x240
	
	subs x8, x8, 0x1
	b.ne  loop_channel_6

activation_6:
	cmp  w7,0
	blt	save_result_6

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

	beq   save_result_6

	dup   v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	
save_result_6:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	sub x13, x13, x14, LSL 4
	add x13, x13, 0x4
	
	mul x10, x14, x4
	sub x11, x11, x10
	add x11, x11, 0x10
	add x12, x2, 0x0

loop_middle_row_end:
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]

	subs x6, x6, 0x1
	b.ne  loop_middle_row

//-------------------------------------------------------- last row 	--------------------------------------------------
	
	
row_first_7:
	// biases_initial
	cbz	x0, none_biases7
		ldr	q16, [x0], 0x10
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q20, [x0], 0x10
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q24, [x0], 0x10
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		ldr	q28, [x0], 0x10
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
		sub  x0, x0, 0x40
	b	convolution_start7
	
none_biases7:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start7:
	add  x8, x4, 0x0
	
loop_channel_7:
	ldr q0, [x11] 
	ldr q1,	[x11, 0x10]
	ldp	q4, q5, [x12]		// k0[7-0]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	subs  x8, x8, 0x1
	fmla	v17.4s, v4.4s,  v0.s[0]		// k0[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k0[f-c]i[3]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k1[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x80]		// k2[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k1[f-c]i[3]
	ldp	q6, q7, [x12,0xa0]		// k2[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k2[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k2[f-c]i[3]
	lsl	x10, x5 ,0x2
	ldr	q0, [x11,x10]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	
	fmla	v17.4s, v4.4s,  v0.s[0]		//k3[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[1]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[2]		// k[3-0]i[3]
	fmla	v21.4s, v5.4s,  v0.s[0]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[1]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[2]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v25.4s, v6.4s,  v0.s[0]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[1]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[2]		// k[b-8]i[3]
	fmla	v29.4s, v7.4s,  v0.s[0]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[1]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[2]		// k3[f-c]i[3]
	ldp	 q6, q7, [x12,0x120]		// k4[f-8]
	add  x10, x10 ,0x10
	ldr	 q1, [x11,x10]
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k7[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k4[f-c]i[3]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k8[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k5[f-c]i[3]
	
	add x11, x11, x14
	add x12, x12, 0x240
	
	b.ne  loop_channel_7
	
activation_7:
    cmp  w7,0
	blt	save_result_7

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s

	beq     save_result_7

	dup     v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s
	
save_result_7:
	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x10
	mul  x10, x14, x4
	sub  x11, x11, x10
	add  x11, x11, 0xc
	add  x12, x2, 0x0		// kernel
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	
row_loop4_8:
	
	lsr  x9, x15, 0x2
	cbz  x9, loop_col_less_8
	
loop_col_8:
	// biases_initial
	cbz	x0, none_biases8
		ldr	q16, [x0], 0x10
		mov v17.16b, v16.16b
		mov v18.16b, v16.16b
		mov v19.16b, v16.16b
		ldr	q20, [x0], 0x10
		mov v21.16b, v20.16b
		mov v22.16b, v20.16b
		mov v23.16b, v20.16b
		ldr	q24, [x0], 0x10
		mov v25.16b, v24.16b
		mov v26.16b, v24.16b
		mov v27.16b, v24.16b
		ldr	q28, [x0], 0x10
		mov v29.16b, v28.16b
		mov v30.16b, v28.16b
		mov v31.16b, v28.16b
		sub  x0, x0, 0x40
	b	convolution_start8
	
none_biases8:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0
	
convolution_start8:
	add  x8, x4, 0x0		// input_chan

loop_channel_8:
	ldr q0, [x11]
	ldr q1, [x11,0x10]
	
	ldp	q4, q5, [x12]			// k0[7-0]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k0[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k0[f-c]i[3]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k1[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x80]		// k2[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k1[f-c]i[3]
	ldp	q6, q7, [x12,0xa0]		// k2[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k2[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	lsl	x10, x5 ,0x2
	ldr	q0, [x11,x10]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k2[f-c]i[3]
	add x10, x10 ,0x10
	ldr	q1, [x11,x10]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	
	fmla	v16.4s, v4.4s,  v0.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[1]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[2]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v0.s[3]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[1]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[2]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v0.s[3]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v24.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[1]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[2]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v0.s[3]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[0]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[1]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[2]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v0.s[3]		// k3[f-c]i[3]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[2]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v0.s[3]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[0]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[2]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v0.s[3]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[0]		// k[7-4]i[3]
	ldp	q4, q5, [x12,0x140]		// k5[7-0]
	fmla	v24.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[2]		// k[b-8]i[1]
	fmla	v26.4s, v6.4s,  v0.s[3]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[0]		// k[b-8]i[3]
	fmla	v28.4s, v7.4s,  v0.s[1]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[2]		// k[f-c]i[1]
	fmla	v30.4s, v7.4s,  v0.s[3]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[0]		// k4[f-c]i[3]
	ldp	q6, q7, [x12,0x160]		// k5[f-8]
	
	fmla	v16.4s, v4.4s,  v0.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v4.4s,  v0.s[3]		// k[3-0]i[1]
	fmla	v18.4s, v4.4s,  v1.s[0]		// k[3-0]i[2]
	fmla	v19.4s, v4.4s,  v1.s[1]		// k[3-0]i[3]
	fmla	v20.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	fmla	v21.4s, v5.4s,  v0.s[3]		// k[7-4]i[1]
	fmla	v22.4s, v5.4s,  v1.s[0]		// k[7-4]i[2]
	fmla	v23.4s, v5.4s,  v1.s[1]		// k[7-4]i[3]
	add  x11, x11, x14
	add  x12, x12, 0x240
	fmla	v24.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v25.4s, v6.4s,  v0.s[3]		// k[b-8]i[1]
	fmla	v28.4s, v7.4s,  v0.s[2]		// k[f-c]i[0]
	fmla	v29.4s, v7.4s,  v0.s[3]		// k[f-c]i[1]
	fmla	v26.4s, v6.4s,  v1.s[0]		// k[b-8]i[2]
	fmla	v27.4s, v6.4s,  v1.s[1]		// k[b-8]i[3]
	fmla	v30.4s, v7.4s,  v1.s[0]		// k[f-c]i[2]
	fmla	v31.4s, v7.4s,  v1.s[1]		// k5[f-c]i[3]
	subs x8, x8, 0x1
	b.ne  loop_channel_8
	
activation_8:
    cmp  w7,0
	blt	save_result_8

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s
	
	beq save_result_8

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s

save_result_8:
	st4 {v16.s,v17.s,v18.s,v19.s}[0], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[1], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[2], [x13], x14
	st4 {v16.s,v17.s,v18.s,v19.s}[3], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[0], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[1], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[2], [x13], x14
	st4 {v20.s,v21.s,v22.s,v23.s}[3], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[0], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[1], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[2], [x13], x14
	st4 {v24.s,v25.s,v26.s,v27.s}[3], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[0], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[1], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[2], [x13], x14
	st4 {v28.s,v29.s,v30.s,v31.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x10
	
loop_col_end_8:
	mul  x10, x14, x4 
	sub  x11, x11, x10
	add  x11, x11, 0x10
	add  x12, x2, 0x0
	
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	subs x9, x9, 0x1
	
	b.ne  loop_col_8
	
loop_col_less_8:
	and x9, x15, 0x3
	cbz x9, row_end_9
	
loop_col_less_start_8:
	add  x8, x4, 0x0
	cbz	x0, none_biases8_1
		ldr	q16, [x0]
		ldr	q17, [x0,0x10]
		ldr	q18, [x0,0x20]
		ldr	q19, [x0,0x30]
	b	loop_channel_8_1
	
none_biases8_1:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
loop_channel_8_1:
	ldr q0, [x11]
	lsl	x10, x5 ,0x2
	ldr q1, [x11, x10]
	ldp q4, q5, [x12, 0x00]		// k0[7-0]
	ldp q6, q7, [x12, 0x20]		// k0[f-8]
	fmla	v16.4s, v4.4s,  v0.s[0]		// k0[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12, 0x40]		// k1[7-0]
	fmla	v18.4s, v6.4s,  v0.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[0]		// k0[f-c]i[0]
	ldp	q6, q7, [x12, 0x60]		// k1[f-8]
	fmla	v16.4s, v4.4s,  v0.s[1]		// k1[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12, 0x80]		// k2[7-0]
	fmla	v18.4s, v6.4s,  v0.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[1]		// k1[f-c]i[0]
	ldp	q6, q7, [x12, 0xa0]		// k2[f-8]
	fmla	v16.4s, v4.4s,  v0.s[2]		// k2[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12, 0xc0]		// k3[7-0]
	fmla	v18.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[2]		// k2[f-c]i[0]
	ldp	q6, q7, [x12, 0xe0]		// k3[f-8]
	
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	fmla	v16.4s, v4.4s,  v1.s[0]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[0]		// k[7-4]i[0]
	ldp	q4, q5, [x12, 0x100]	// k4[7-0]
	fmla	v18.4s, v6.4s,  v1.s[0]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[0]		// k3[f-c]i[0]
	ldp	q6, q7, [x12, 0x120]	// k4[f-8]
	fmla	v16.4s, v4.4s,  v1.s[1]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[1]		// k[7-4]i[0]
	ldp	q4, q5, [x12, 0x140]	// k5[7-0]
	fmla	v18.4s, v6.4s,  v1.s[1]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[1]		// k4[f-c]i[0]
	ldp	q6, q7, [x12, 0x160]	// k5[f-8]
	fmla	v16.4s, v4.4s,  v1.s[2]		// k5[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[2]		// k[7-4]i[0]
	fmla	v18.4s, v6.4s,  v1.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[2]		// k5[f-c]i[0]
	add x11, x11, x14
	add x12, x12, 0x240
	subs x8, x8, 0x1
	b.ne  loop_channel_8_1
	
activation_8_1:
	cmp w7,0
	blt	save_result_8_1

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

	beq save_result_8_1

	dup  v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	
save_result_8_1:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	sub  x13, x13, x14, LSL 4
	add  x13, x13, 0x4
	
loop_col_less_end_8:
	mul  x10, x14, x4 
	sub  x11, x11, x10
	add  x11, x11, 0x4
	add  x12, x2, 0x0
	
	prfm	pldl1keep, [x12]
	prfm	pldl1keep, [x11]
	subs x9, x9, 0x1
	b.ne  loop_col_less_start_8
	

row_end_9:
	cbz	x0, none_biases9
		ldr	q16, [x0]
		ldr	q17, [x0,0x10]
		ldr	q18, [x0,0x20]
		ldr	q19, [x0,0x30]
	b	convolution_start9
	
none_biases9:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	
convolution_start9:
	add  x8,  x4, 0x0
	sub  x11, x11, 0x8
	lsl	 x10, x5 ,0x2
	
loop_channel_9:
	ldr q0, [x11] 
	ldr q1,	[x11, x10]
	ldp	q4, q5, [x12,0x00]		// k0[7-0]
	ldp	q6, q7, [x12,0x20]		// k0[f-8]
	fmla	v16.4s, v4.4s,  v0.s[2]		// k0[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x40]		// k1[7-0]
	fmla	v18.4s, v6.4s,  v0.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[2]		// k0[f-c]i[0]
	ldp	q6, q7, [x12,0x60]		// k1[f-8]
	fmla	v16.4s, v4.4s,  v0.s[3]		// k1[3-0]i[0]
	fmla	v17.4s, v5.4s,  v0.s[3]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0xc0]		// k3[7-0]
	fmla	v18.4s, v6.4s,  v0.s[3]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v0.s[3]		// k1[f-c]i[0]
	ldp	q6, q7, [x12,0xe0]		// k3[f-8]
	
	prfm	pldl1keep, [x12, 0x240]
	prfm	pldl1keep, [x11, x14]
	fmla	v16.4s, v4.4s,  v1.s[2]		// k3[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[2]		// k[7-4]i[0]
	ldp	q4, q5, [x12,0x100]		// k4[7-0]
	fmla	v18.4s, v6.4s,  v1.s[2]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[2]		// k3[f-c]i[0]
	ldp	q6, q7, [x12,0x120]		// k4[f-8]
	fmla	v16.4s, v4.4s,  v1.s[3]		// k4[3-0]i[0]
	fmla	v17.4s, v5.4s,  v1.s[3]		// k[7-4]i[0]
	subs x8, x8, 0x1
	fmla	v18.4s, v6.4s,  v1.s[3]		// k[b-8]i[0]
	fmla	v19.4s, v7.4s,  v1.s[3]		// k4[f-c]i[0]
	add x11, x11, x14
	add x12, x12, 0x240
	
	b.ne  loop_channel_9

activation_9:
	cmp w7,0
	blt	save_result_9

	movi	d0, 0
    scvtf   s1,w7

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s

	beq     save_result_9

	dup    v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	
save_result_9:
	st1 {v16.s}[0], [x13], x14
	st1 {v16.s}[1], [x13], x14
	st1 {v16.s}[2], [x13], x14
	st1 {v16.s}[3], [x13], x14
	st1 {v17.s}[0], [x13], x14
	st1 {v17.s}[1], [x13], x14
	st1 {v17.s}[2], [x13], x14
	st1 {v17.s}[3], [x13], x14
	st1 {v18.s}[0], [x13], x14
	st1 {v18.s}[1], [x13], x14
	st1 {v18.s}[2], [x13], x14
	st1 {v18.s}[3], [x13], x14
	st1 {v19.s}[0], [x13], x14
	st1 {v19.s}[1], [x13], x14
	st1 {v19.s}[2], [x13], x14
	st1 {v19.s}[3], [x13], x14
	

	ret


	.end


