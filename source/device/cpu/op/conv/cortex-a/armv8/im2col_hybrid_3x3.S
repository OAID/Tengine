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
// im2col for kernel 3x3 d1  include 2 function  stride 1 and stride 2
//
// input:
//         x0 arg0  input address 
//         x1 arg1  input_x
//         x2 arg2  input_y
//         x3 arg3  input channel cnt
//         x4 arg4  col address
//         x5 arg5  stride_x
//         x6 arg6  scale address
//
// register definition
//    x0 c0l0 address  q0  q1  q2 
//    x1 input_x x 4
//    x2 input_xy x 4
//    x3 input channel
//    x4 col address
//    x5 stride_x
//    x6 scale address
//    x7 channel cnt
//    x11 c0l1 address q3  q4  q5
//    x12 c0l2 address q16 q17 q18
//    x13 c1l0 address q19 q20 q21
//    x14 c1l1 address q22 q23 q24
//    x15 c1l2 address q25 q26 q27

        .section .text,"ax"
        .align 5

        .type   im2col_hybrid_3x3 STT_FUNC
        .global im2col_hybrid_3x3
        .hidden im2col_hybrid_3x3
im2col_hybrid_3x3:
	// initial
	ldr	s31,[x6]
	lsl	x1, x1, 2	// x1 = input_x size
	mul	x2, x2, x1	// x2 = input_xy size
	
	add	x11,x0, x1
	add	x12,x0, x1, LSL 1

	cmp	x5, 2
	beq	stride2

stride1:
	cmp	x3, 2
	blt	stride1_channel_loop_end
	lsr	x7, x3, 1

	add	x13,x0, x2
	add	x14,x11,x2
	add	x15,x12,x2

stride1_channel_loop2:
	ldr	q0,  [x0]	
	ldr	q3,  [x11]
	ldr	d1,  [x0, 0x10]	
	ldr	d4,  [x11,0x10]
	ldr	q16, [x12]
	subs	x7, x7, 1
	ldr	q19, [x13]
	fmul	v0.4s, v0.4s, v31.s[0]
	ldr	d17, [x12,0x10]
	fmul	v3.4s, v3.4s, v31.s[0]
	ldr	d20, [x13,0x10]
	fmul	v1.2s, v1.2s, v31.s[0]
	ldr	q22, [x14]
	fmul	v4.2s, v4.2s, v31.s[0]
	ldr	q25, [x15]
	fmul	v16.4s,v16.4s,v31.s[0]
	ldr	d23, [x14,0x10]
	fmul	v19.4s,v19.4s,v31.s[0]
	ldr	d26, [x15,0x10]
	fmul	v17.2s,v17.2s,v31.s[0]

	prfm	pldl1strm, [x0, 0x40]
	add	x0, x0, x2, LSL 1
	fmul	v20.2s,v20.2s,v31.s[0]
	prfm	pldl1strm, [x11,0x40]
	add	x11,x11,x2, LSL 1
	fmul	v22.4s,v22.4s,v31.s[0]
	prfm	pldl1strm, [x12,0x40]
	add	x12,x12,x2, LSL 1
	fmul	v25.4s,v25.4s,v31.s[0]
	prfm	pldl1strm, [x13,0x40]
	add	x13,x13,x2, LSL 1
	fmul	v23.2s,v23.2s,v31.s[0]
	prfm	pldl1strm, [x14,0x40]
	add	x14,x14,x2, LSL 1
	fmul	v26.2s,v26.2s,v31.s[0]
	prfm	pldl1strm, [x15,0x40]
	add	x15,x15,x2, LSL 1

	fcvtas	v0.4s, v0.4s
	st1	{v0.b}[0],  [x4], 1
	fcvtas	v3.4s, v3.4s
	st1	{v0.b}[4],  [x4], 1
	fcvtas	v1.2s, v1.2s
	st1	{v0.b}[4],  [x4], 1
	fcvtas	v4.2s, v4.2s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v16.4s, v16.4s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v19.4s, v19.4s
	st1	{v0.b}[12], [x4], 1
	fcvtas	v17.2s, v17.2s
	st1	{v0.b}[12], [x4], 1
	fcvtas	v20.2s, v20.2s
	st1	{v1.b}[0],  [x4], 1
	fcvtas	v22.4s, v22.4s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v25.4s, v25.4s
	st1	{v3.b}[0],  [x4], 1
	fcvtas	v23.2s, v23.2s
	st1	{v0.b}[12], [x4], 1
	fcvtas	v26.2s, v26.2s
	
	st1	{v3.b}[4],  [x4], 1
	st1	{v1.b}[0],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v1.b}[4],  [x4], 1
	st1	{v3.b}[12], [x4], 1

	st1	{v3.b}[4],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[4],  [x4], 1

	st1	{v16.b}[0], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v17.b}[0], [x4], 1

	st1	{v16.b}[8], [x4], 1
	st1	{v19.b}[0], [x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v19.b}[4], [x4], 1
	st1	{v17.b}[0], [x4], 1
	st1	{v19.b}[8], [x4], 1
	st1	{v17.b}[4], [x4], 1
	st1	{v19.b}[12],[x4], 1

	st1	{v19.b}[4], [x4], 1
	st1	{v19.b}[8], [x4], 1
	st1	{v19.b}[8], [x4], 1
	st1	{v19.b}[12],[x4], 1
	st1	{v19.b}[12],[x4], 1
	st1	{v20.b}[0], [x4], 1
	st1	{v20.b}[0], [x4], 1
	st1	{v20.b}[4], [x4], 1

	st1	{v22.b}[0], [x4], 1
	st1	{v22.b}[4], [x4], 1
	st1	{v22.b}[4], [x4], 1
	st1	{v22.b}[8], [x4], 1
	st1	{v22.b}[8], [x4], 1
	st1	{v22.b}[12],[x4], 1
	st1	{v22.b}[12],[x4], 1
	st1	{v23.b}[0], [x4], 1

	st1	{v22.b}[8], [x4], 1
	st1	{v25.b}[0], [x4], 1
	st1	{v22.b}[12],[x4], 1
	st1	{v25.b}[4], [x4], 1
	st1	{v23.b}[0], [x4], 1
	st1	{v25.b}[8], [x4], 1
	st1	{v23.b}[4], [x4], 1
	st1	{v25.b}[12],[x4], 1

	st1	{v25.b}[4], [x4], 1
	st1	{v25.b}[8], [x4], 1
	st1	{v25.b}[8], [x4], 1
	st1	{v25.b}[12],[x4], 1
	st1	{v25.b}[12],[x4], 1
	st1	{v26.b}[0], [x4], 1
	st1	{v26.b}[0], [x4], 1
	st1	{v26.b}[4], [x4], 1
	bne	stride1_channel_loop2


stride1_channel_loop_end:
	and	x3, x3, 0x1
	cbz	x3, finish

	ldr	q0,  [x0]	
	ldr	d1,  [x0, 0x10]	
	ldr	q3,  [x11]
	ldr	d4,  [x11,0x10]
	ldr	q16, [x12]
	ldr	d17, [x12,0x10]

	fmul	v0.4s, v0.4s, v31.s[0]
	prfm	pldl1strm, [x0,0x40]
	fmul	v3.4s, v3.4s, v31.s[0]
	prfm	pldl1strm, [x11,0x40]
	fmul	v1.2s, v1.2s, v31.s[0]
	fmul	v4.2s, v4.2s, v31.s[0]
	fmul	v16.4s,v16.4s,v31.s[0]
	prfm	pldl1strm, [x12,0x40]
	fmul	v17.2s,v17.2s,v31.s[0]

	fcvtas	v0.4s, v0.4s
	fcvtas	v3.4s, v3.4s
	fcvtas	v1.2s, v1.2s
	st1	{v0.b}[0],  [x4], 1
	fcvtas	v4.2s, v4.2s
	st1	{v0.b}[4],  [x4], 1
	fcvtas	v16.4s, v16.4s
	st1	{v0.b}[4],  [x4], 1
	fcvtas	v17.2s, v17.2s

	st1	{v0.b}[8],  [x4], 1
	st1	{v0.b}[8],  [x4], 1
	st1	{v0.b}[12], [x4], 1
	st1	{v0.b}[12], [x4], 1
	st1	{v1.b}[0],  [x4], 1
	
	st1	{v0.b}[8],  [x4], 1
	st1	{v3.b}[0],  [x4], 1
	st1	{v0.b}[12], [x4], 1
	st1	{v3.b}[4],  [x4], 1
	st1	{v1.b}[0],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v1.b}[4],  [x4], 1
	st1	{v3.b}[12], [x4], 1

	st1	{v3.b}[4],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[4],  [x4], 1

	st1	{v16.b}[0], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v17.b}[0], [x4], 1

	st1	{v16.b}[8], [x4], 1
	strb	wzr,        [x4], 1
	st1	{v16.b}[12],[x4], 1
	strb	wzr,        [x4], 1
	st1	{v17.b}[0], [x4], 1
	strb	wzr,        [x4], 1
	st1	{v17.b}[4], [x4], 1
	strb	wzr,        [x4]

	b	finish

stride2:
	cmp	x3, 2
	blt	stride2_channel_loop_end
	lsr	x7, x3, 1

	add	x13,x0, x2
	add	x14,x11,x2
	add	x15,x12,x2

stride2_channel_loop2:
	ldp	q0, q1, [x0]	
	ldr	s2, [x0, 0x20]	
	subs	x7, x7, 1
	ldp	q3, q4, [x11]	
	ldr	s5, [x11,0x20]	
	fmul	v0.4s, v0.4s, v31.s[0]
	ldp	q16,q17,[x12]	
	fmul	v1.4s, v1.4s, v31.s[0]
	ldr	s18,[x12,0x20]	
	fmul	s2, s2, s31
	ldp	q19,q20,[x13]	
	fmul	v3.4s, v3.4s, v31.s[0]
	ldr	s21,[x13,0x20]	
	fmul	v4.4s, v4.4s, v31.s[0]
	ldp	q22,q23,[x14]	
	fmul	s5, s5, s31
	ldr	s24,[x14,0x20]	
	fmul	v16.4s,v16.4s,v31.s[0]
	ldp	q25,q26,[x15]	
	fmul	v17.4s,v17.4s,v31.s[0]
	ldr	s27,[x15,0x20]	

	fmul	s18, s18, s31
	prfm	pldl1strm, [x0, 0x60]
	fmul	v19.4s,v19.4s,v31.s[0]
	add	x0, x0, x2, LSL 1
	fmul	v20.4s,v20.4s,v31.s[0]
	prfm	pldl1strm, [x11,0x60]
	fmul	s21, s21 ,s31
	add	x11,x11,x2, LSL 1
	fmul	v22.4s,v22.4s,v31.s[0]
	prfm	pldl1strm, [x12,0x60]
	fmul	v23.4s,v23.4s,v31.s[0]
	add	x12,x12,x2, LSL 1
	fmul	s24, s24, s31
	prfm	pldl1strm, [x13,0x60]
	fmul	v25.4s,v25.4s,v31.s[0]
	add	x13,x13,x2, LSL 1
	fmul	v26.4s,v26.4s,v31.s[0]
	prfm	pldl1strm, [x14,0x60]
	fmul	s27, s27, s31
	add	x14,x14,x2, LSL 1

	fcvtas	v0.4s, v0.4s
	prfm	pldl1strm, [x15,0x60]
	fcvtas	v1.4s, v1.4s
	add	x15,x15,x2, LSL 1
	fcvtas	v2.2s, v2.2s
	st1	{v0.b}[0],  [x4], 1
	fcvtas	v3.4s, v3.4s
	st1	{v0.b}[4],  [x4], 1
	fcvtas	v4.4s, v4.4s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v5.2s, v5.2s
	st1	{v0.b}[12], [x4], 1
	fcvtas	v16.4s, v16.4s
	st1	{v1.b}[0],  [x4], 1
	fcvtas	v17.4s, v17.4s
	st1	{v1.b}[4],  [x4], 1
	fcvtas	v18.2s, v18.2s
	st1	{v1.b}[8],  [x4], 1
	fcvtas	v19.4s, v19.4s
	st1	{v1.b}[12], [x4], 1
	fcvtas	v20.4s, v20.4s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v21.2s, v21.2s
	st1	{v3.b}[0],  [x4], 1
	fcvtas	v22.4s, v22.4s
	st1	{v1.b}[0],  [x4], 1	
	fcvtas	v23.4s, v23.4s
	st1	{v3.b}[8],  [x4], 1
	fcvtas	v24.2s, v24.2s
	st1	{v1.b}[8],  [x4], 1
	fcvtas	v25.4s, v25.4s
	st1	{v4.b}[0],  [x4], 1
	fcvtas	v26.4s, v26.4s
	st1	{v2.b}[0],  [x4], 1
	fcvtas	v27.2s, v27.2s
	st1	{v4.b}[8],  [x4], 1

	st1	{v3.b}[4],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[4],  [x4], 1
	st1	{v4.b}[8],  [x4], 1
	st1	{v4.b}[12], [x4], 1
	st1	{v5.b}[0],  [x4], 1

	st1	{v16.b}[0], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v17.b}[0], [x4], 1
	st1	{v17.b}[4], [x4], 1
	st1	{v17.b}[8], [x4], 1
	st1	{v17.b}[12],[x4], 1

	st1	{v16.b}[8], [x4], 1
	st1	{v19.b}[0], [x4], 1
	st1	{v17.b}[0], [x4], 1
	st1	{v19.b}[8], [x4], 1
	st1	{v17.b}[8], [x4], 1
	st1	{v20.b}[0], [x4], 1
	st1	{v18.b}[0], [x4], 1
	st1	{v20.b}[8], [x4], 1

	st1	{v19.b}[4], [x4], 1
	st1	{v19.b}[8], [x4], 1
	st1	{v19.b}[12],[x4], 1
	st1	{v20.b}[0], [x4], 1
	st1	{v20.b}[4], [x4], 1
	st1	{v20.b}[8], [x4], 1
	st1	{v20.b}[12],[x4], 1
	st1	{v21.b}[0], [x4], 1

	st1	{v22.b}[0], [x4], 1
	st1	{v22.b}[4], [x4], 1
	st1	{v22.b}[8], [x4], 1
	st1	{v22.b}[12],[x4], 1
	st1	{v23.b}[0], [x4], 1
	st1	{v23.b}[4], [x4], 1
	st1	{v23.b}[8], [x4], 1
	st1	{v23.b}[12],[x4], 1

	st1	{v22.b}[8], [x4], 1
	st1	{v25.b}[0], [x4], 1
	st1	{v23.b}[0], [x4], 1
	st1	{v25.b}[8], [x4], 1
	st1	{v23.b}[8], [x4], 1
	st1	{v26.b}[0], [x4], 1
	st1	{v24.b}[0], [x4], 1
	st1	{v26.b}[8], [x4], 1

	st1	{v25.b}[4], [x4], 1
	st1	{v25.b}[8], [x4], 1
	st1	{v25.b}[12],[x4], 1
	st1	{v26.b}[0], [x4], 1
	st1	{v26.b}[4], [x4], 1
	st1	{v26.b}[8], [x4], 1
	st1	{v26.b}[12],[x4], 1
	st1	{v27.b}[0], [x4], 1
	bne	stride2_channel_loop2

stride2_channel_loop_end:
	and	x3, x3, 0x1
	cbz	x3, finish

	ldp	q0, q1, [x0]	
	ldr	s2, [x0, 0x20]	
	ldp	q3, q4, [x11]	
	ldr	s5, [x11,0x20]	
	fmul	v0.4s, v0.4s, v31.s[0]
	ldp	q16,q17,[x12]	
	fmul	v1.4s, v1.4s, v31.s[0]
	ldr	s18,[x12,0x20]	

	fmul	v2.2s, v2.2s, v31.s[0]
	prfm	pldl1strm, [x0, 0x60]
	fmul	v3.4s, v3.4s, v31.s[0]
	prfm	pldl1strm, [x11,0x60]
	fmul	v4.4s, v4.4s, v31.s[0]
	prfm	pldl1strm, [x12,0x60]
	fmul	v5.2s, v5.2s, v31.s[0]
	fcvtas	v0.4s, v0.4s
	fmul	v16.4s,v16.4s,v31.s[0]
	fcvtas	v1.4s, v1.4s
	fmul	v17.4s,v17.4s,v31.s[0]
	st1	{v0.b}[0],  [x4], 1
	fmul	v18.2s,v18.2s,v31.s[0]
	st1	{v0.b}[4],  [x4], 1

	fcvtas	v2.2s, v2.2s
	st1	{v0.b}[8],  [x4], 1
	fcvtas	v3.4s, v3.4s
	st1	{v0.b}[12], [x4], 1
	fcvtas	v4.4s, v4.4s
	st1	{v1.b}[0],  [x4], 1
	fcvtas	v5.2s, v5.2s
	st1	{v1.b}[4],  [x4], 1
	fcvtas	v16.4s, v16.4s
	st1	{v1.b}[8],  [x4], 1
	fcvtas	v17.4s, v17.4s
	st1	{v1.b}[12], [x4], 1
	fcvtas	v18.2s, v18.2s

	st1	{v0.b}[8],  [x4], 1
	st1	{v3.b}[0],  [x4], 1
	st1	{v1.b}[0],  [x4], 1	
	st1	{v3.b}[8],  [x4], 1
	st1	{v1.b}[8],  [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v2.b}[0],  [x4], 1
	st1	{v4.b}[8],  [x4], 1

	st1	{v3.b}[4],  [x4], 1
	st1	{v3.b}[8],  [x4], 1
	st1	{v3.b}[12], [x4], 1
	st1	{v4.b}[0],  [x4], 1
	st1	{v4.b}[4],  [x4], 1
	st1	{v4.b}[8],  [x4], 1
	st1	{v4.b}[12], [x4], 1
	st1	{v5.b}[0],  [x4], 1

	st1	{v16.b}[0], [x4], 1
	st1	{v16.b}[4], [x4], 1
	st1	{v16.b}[8], [x4], 1
	st1	{v16.b}[12],[x4], 1
	st1	{v17.b}[0], [x4], 1
	st1	{v17.b}[4], [x4], 1
	st1	{v17.b}[8], [x4], 1
	st1	{v17.b}[12],[x4], 1

	st1	{v16.b}[8], [x4], 1
	strb	wzr,        [x4], 1
	st1	{v17.b}[0], [x4], 1
	strb	wzr,        [x4], 1
	st1	{v17.b}[8], [x4], 1
	strb	wzr,        [x4], 1
	st1	{v18.b}[0], [x4], 1
	strb	wzr,        [x4]

finish:

	ret


	.end
