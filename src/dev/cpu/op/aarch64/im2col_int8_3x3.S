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
//    x0 c0h0 address  q0  q1  q2 
//    x1 input_x x 4
//    x2 input_xy x 4
//    x3 input channel
//    x4 col address
//    x5 stride_x
//    x6 scale address
//    x7 channel cnt
//    x11 c0h1 address q3  q4  q5
//    x12 c0h2 address q16 q17 q18
//    x13 c1h0 address q19 q20 q21
//    x14 c1h1 address q22 q23 q24
//    x15 c1h2 address q25 q26 q27

        .section .text,"ax"
        .align 5

        .type   im2col_int8_3x3 STT_FUNC
        .global im2col_int8_3x3
        .hidden im2col_int8_3x3
im2col_int8_3x3:
	// initial
	mul	    x2, x2, x1	// x2 = input_xy size
	
	add	    x11,x0, x1
	add	    x12,x0, x1, LSL 1

	add	    x13,x0, x2
	add	    x14,x11,x2
	add	    x15,x12,x2
    
	lsr	    x7, x3, 1
    
	cmp	    x5, 2
	beq	    stride2

stride1:
	cmp	    x3, 2
	blt	    stride1_channel_loop_end


stride1_channel_loop2:
	ldr	    d0,  [x0]	
	ldr	    d3,  [x11]
	ldr	    d16, [x12]
    ext     v1.8b, v0.8b, v0.8b, 1
    ext     v2.8b, v0.8b, v0.8b, 2
    ext     v4.8b, v3.8b, v3.8b, 1
    ext     v5.8b, v3.8b, v3.8b, 2
    ext     v17.8b, v16.8b, v16.8b, 1
    ext     v18.8b, v16.8b, v16.8b, 2
	subs	x7, x7, 1
	ldr	    d19, [x13]
	ldr	    d22, [x14]
	ldr	    d25, [x15]
    ext     v20.8b, v19.8b, v19.8b, 1
    ext     v21.8b, v19.8b, v19.8b, 2
    ext     v23.8b, v22.8b, v22.8b, 1
    ext     v24.8b, v22.8b, v22.8b, 2
    ext     v26.8b, v25.8b, v25.8b, 1
    ext     v27.8b, v25.8b, v25.8b, 2

	prfm	pldl1strm, [x0, 0x40]
	add	    x0, x0, x2, LSL 1
	prfm	pldl1strm, [x11,0x40]
	add	    x11, x11, x2, LSL 1
	prfm	pldl1strm, [x12,0x40]
	add	    x12, x12, x2, LSL 1
	prfm	pldl1strm, [x13,0x40]
	add	    x13, x13, x2, LSL 1
	prfm	pldl1strm, [x14,0x40]
	add	    x14, x14, x2, LSL 1
	prfm	pldl1strm, [x15,0x40]
	add	    x15, x15, x2, LSL 1

    zip1    v0.8b, v0.8b, v1.8b
    zip1    v2.8b, v2.8b, v3.8b
	st1	    {v0.8b}, [x4], 8
    zip1    v4.8b, v4.8b, v5.8b
	st1	    {v2.8b}, [x4], 8
    zip1    v16.8b, v16.8b, v17.8b
	st1	    {v4.8b}, [x4], 8
    zip1    v18.8b, v18.8b, v19.8b
	st1	    {v16.8b}, [x4], 8
    zip1    v20.8b, v20.8b, v21.8b
	st1	    {v18.8b}, [x4], 8
    zip1    v22.8b, v22.8b, v23.8b
	st1	    {v20.8b}, [x4], 8
    zip1    v24.8b, v24.8b, v25.8b
	st1	    {v22.8b}, [x4], 8
    zip1    v26.8b, v26.8b, v27.8b
    
	st1	    {v24.8b}, [x4], 8
	st1	    {v26.8b}, [x4], 8
	bne	stride1_channel_loop2


stride1_channel_loop_end:
	and	x3, x3, 0x1
	cbz	x3, finish

	ldr	    d0,  [x0]
	ldr	    d3,  [x11]
	ldr	    d16, [x12]
	movi	d19, 0

    ext     v1.8b, v0.8b, v0.8b, 1
    ext     v2.8b, v0.8b, v0.8b, 2
	prfm	pldl1strm, [x0,0x40]
    ext     v4.8b, v3.8b, v3.8b, 1
    ext     v5.8b, v3.8b, v3.8b, 2
	prfm	pldl1strm, [x11,0x40]
    ext     v17.8b, v16.8b, v16.8b, 1
    ext     v18.8b, v16.8b, v16.8b, 2
	prfm	pldl1strm, [x12,0x40]
    
    zip1    v0.8b, v0.8b, v1.8b
    zip1    v2.8b, v2.8b, v3.8b
	st1	    {v0.8b}, [x4], 8
    zip1    v4.8b, v4.8b, v5.8b
	st1	    {v2.8b}, [x4], 8
    zip1    v16.8b, v16.8b, v17.8b
	st1	    {v4.8b}, [x4], 8
    zip1    v18.8b, v18.8b, v19.8b
	st1	    {v16.8b}, [x4], 8
	st1	    {v18.8b}, [x4], 8

	b	    finish

stride2:
	cmp	    x3, 2
	blt	    stride2_channel_loop_end


stride2_channel_loop2:
	ldr	    q0, [x0]
	ldr	    q3, [x11]
	ldr	    q16, [x12]
	ldr	    q19, [x13]
	ldr	    q22, [x14]
	ldr	    q25, [x15]
	subs	x7, x7, 1

	prfm	pldl1strm, [x0, 0x60]
	add	    x0, x0, x2, LSL 1
	prfm	pldl1strm, [x11,0x60]
	add	    x11,x11,x2, LSL 1
	prfm	pldl1strm, [x12,0x60]
	add	    x12,x12,x2, LSL 1
	prfm	pldl1strm, [x13,0x60]
	add	    x13,x13,x2, LSL 1
	prfm	pldl1strm, [x14,0x60]
	add	    x14,x14,x2, LSL 1
	prfm	pldl1strm, [x15,0x60]
	add	    x15,x15,x2, LSL 1
    
    ext     v1.16b, v0.16b, v0.16b, 2
    ext     v4.16b, v3.16b, v3.16b, 1
    trn1    v2.16b, v1.16b, v3.16b
    
    ext     v17.16b, v16.16b, v16.16b, 2
    ext     v20.16b, v19.16b, v19.16b, 1
    trn1    v18.16b, v17.16b, v19.16b
    
    ext     v23.16b, v22.16b, v22.16b, 2
    ext     v26.16b, v25.16b, v25.16b, 1
    trn1    v24.16b, v23.16b, v25.16b

	st1	    {v0.8b}, [x4], 8
	st1	    {v2.8b}, [x4], 8
	st1	    {v4.8b}, [x4], 8
    
	st1	    {v16.8b}, [x4], 8
	st1	    {v18.8b}, [x4], 8
	st1	    {v20.8b}, [x4], 8
    
	st1	    {v22.8b}, [x4], 8
	st1	    {v24.8b}, [x4], 8
	st1	    {v26.8b}, [x4], 8
	bne	stride2_channel_loop2

stride2_channel_loop_end:
	and	x3, x3, 0x1
	cbz	x3, finish
    
    ldr	    q0, [x0]
	ldr	    q3, [x11]
	ldr	    q16, [x12]
	movi	d19, 0
	subs	x7, x7, 1

	prfm	pldl1strm, [x0, 0x60]
    ext     v1.16b, v0.16b, v0.16b, 2
    ext     v4.16b, v3.16b, v3.16b, 1
    trn1    v2.16b, v1.16b, v3.16b
	prfm	pldl1strm, [x11,0x60]
    ext     v17.16b, v16.16b, v16.16b, 2
	prfm	pldl1strm, [x12,0x60]
    trn1    v18.16b, v17.16b, v19.16b

	st1	    {v0.8b}, [x4], 8
	st1	    {v2.8b}, [x4], 8
	st1	    {v4.8b}, [x4], 8
    
	st1	    {v16.8b}, [x4], 8
	st1	    {v18.8b}, [x4], 8

finish:

	ret


	.end
