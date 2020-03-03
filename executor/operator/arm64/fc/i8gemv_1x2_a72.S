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
// 1*2 8bit signed integer matric multiplication
//
//
//                            --           --
//                            |   k0   k1   |
//                            |   .    .    |
//    --              --      |   .    .    |      --          --         --            --
//    | i0 - - - - - - |  x   |   .    .    |   +  |   b0   b1   |    =   |  i0k0  i0k1  |
//    --              --      |   .    .    |      --          --         --            --
//                            |   .    .    |
//                            |   .    .    |
//                            --           --
//      input 1 x p             kernel p x 2          biases x 2            output 1 x 2           p = kernel size
//
//
// optimised for Cortex-A72 pipeline 10 cycle per loop (1*2*16 dot product) but the bottleneck is memory bandwidth
//
// input:
//         x0   arg0   biases start address      {b0, b1}
//         x1   arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg2   kernel data start address {k0[0-1],k1[0-1],k0[2-3],k1[2-3],...}
//         x3   arg3   kernel size need to be even number
//         x4   arg4   output data save address  {ik0, ik1}
//         x5   arg5   scale address
//
// output: no
//
// v0   8byte data of input { i[ 7- 0]} 
// v1   8byte data of input { i[15- 8]}
// V2 ~ V3 not used 
// v4  16byte data of input { i[ 7- 6], i[ 7- 6], i[ 5- 4], i[ 5- 4], i[ 3- 2], i[ 3- 2], i[ 1- 0], i[ 1- 0]} 
// v5  16byte data of input { i[15-14], i[15-14], i[13-12], i[13-12], i[11-10], i[11-10], i[ 9- 8], i[ 9- 8]} 
// v6 ~ v15 not used
// v16 16byte data of kernel{k1[ 7- 6],k0[ 7- 6],k1[ 5- 4],k0[ 5- 4],k1[ 3- 2],k0[ 3- 2],k1[ 1- 0],k0[ 1- 0]}
// v17 16byte data of kernel{k1[15-14],k0[15-14],k1[13-12],k0[13-12],k1[13-12],k0[11-10],k1[11-10],k0[ 9- 8]}
// v18 ~ V23 not used 
// v24 ~ V25 temp register
// V26 ~ V29 not used
// v30 dot product for  {ik1,  ik0, ik1, ik0}
// V31 not used

        .section .text,"ax"
        .align 5

        .type i8gemv_1x2_a72 STT_FUNC
        .global i8gemv_1x2_a72
        .hidden i8gemv_1x2_a72
i8gemv_1x2_a72:
// initial
	cmp     x3, 16 
	and     x10, x3, 0x3
	movi	d30, 0

	b.lt	last8
	lsr	x9, x3, 0x4		// x9 = kernel_size / 16
	and	x3, x3, 0xf
	
// main loop     each loop generate dot prodcut for 1x2x16 data
loop16:  
	ldp	d0, d1, [x1]		// d0  = i[7-0]    d1 = i[15-8]
	ldr	q16,[x2]		// q16 = k1[ 7- 6],k0[ 7- 6],k1[ 5- 4],k0[ 5- 4],k1[ 3- 2],k0[ 3- 2],k1[ 1- 0],k0[ 1- 0]
	ldr	q17,[x2, 0x10]		// q16 = k1[ 7- 6],k0[ 7- 6],k1[ 5- 4],k0[ 5- 4],k1[ 3- 2],k0[ 3- 2],k1[ 1- 0],k0[ 1- 0]
	subs	x9, x9, 1		// loop counter
					// q17 = k1[15-14],k0[15-14],k1[13-12],k0[13-12],k1[13-12],k0[11-10],k1[11-10],k0[ 9- 8]
	zip1	v4.8h, v0.8h, v0.8h	// q4  =  i[ 7- 6], i[ 7- 6], i[ 5- 4], i[ 5- 4], i[ 3- 2], i[ 3- 2], i[ 1- 0], i[ 1- 0]
	zip1	v5.8h, v1.8h, v1.8h	// q5  =  i[15-14], i[15-14], i[13-12], i[13-12], i[11-10], i[11-10], i[ 9- 8], i[ 9- 8] 
	prfm	pldl1keep, [x1, 0xb0]
	prfm	pldl1keep, [x2, 0x180]
	
	smull	v24.8h, v16.8b, v4.8b	// ik[1-0][ 3- 0]
	add	x1, x1, 0x10
	smlal2	v24.8h, v16.16b,v4.16b  // ik[1-0][ 7- 4]
	add	x2, x2, 0x20
	smull	v25.8h, v17.8b, v5.8b	// ik[1-0][11- 8]
	smlal2	v25.8h, v17.16b,v5.16b  // ik[1-0][15-12]
	sadalp	v30.4s, v24.8h
	sadalp	v30.4s, v25.8h

	b.ne	loop16

last8:
	cmp	x3, 0x8
	b.lt	last4
	and	x3, x3, 0x7

	ldr	d0, [x1], 0x8		// d0  = i[7-0]
	ldr	q16,[x2], 0x10		// q16 = k1[ 7- 6],k0[ 7- 6],k1[ 5- 4],k0[ 5- 4],k1[ 3- 2],k0[ 3- 2],k1[ 1- 0],k0[ 1- 0]
	zip1	v4.8h, v0.8h, v0.8h	// v4  =  i[ 7- 6], i[ 7- 6], i[ 5- 4], i[ 5- 4], i[ 3- 2], i[ 3- 2], i[ 1- 0], i[ 1- 0]
	smull	v24.8h, v16.8b, v4.8b	// ik[1-0][ 3- 0]
	smlal2	v24.8h, v16.16b,v4.16b  // ik[1-0][ 7- 4]
	sadalp	v30.4s, v24.8h

last4:
	cmp	x3, 0x4
	b.lt	last2

	ldr	s0, [x1], 0x4		// s0  = i[3-0]
	ldr	d16,[x2], 0x8		// d16 = k1[3-2],k0[3-2],k1[1-0],k0[1-0]
	zip1	v4.4h, v0.4h, v0.4h	// v4  = i[ 3- 2], i[ 3- 2], i[ 1- 0], i[ 1- 0]
	smull	v24.8h, v16.8b, v4.8b	// ik[1-0][ 3- 0]
	sadalp	v30.4s, v24.8h

last2:
	cbz	x10, int2fp

	ld1r	{v4.4h},[x1]		// v4  =  i[ 1- 0], i[ 1- 0]
	ldr	s16,[x2]		// s16 = k1[ 1- 0],k0[ 1- 0]
	smull	v24.8h, v16.8b, v4.8b	// ik[1-0][ 3- 0]
	sadalp	v30.4s, v24.8h

int2fp:
	// convert result to sp and multiply with scale
        ldr     s0, [x5]
        scvtf   v30.4s, v30.4s
	mov	v1.d[0], v30.d[1]
	fadd	v30.2s, v30.2s, v1.2s
	fmul    v30.2s, v30.2s, v0.s[0]

// load and add biases
	cbz	x0, save_result
	ldr	d0, [x0]
	fadd	v30.2s, v30.2s, v0.2s

save_result:
	str	d30, [x4]

	ret

        .end
