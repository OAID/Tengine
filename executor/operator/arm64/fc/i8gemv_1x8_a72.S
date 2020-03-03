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
// 1*8 8bit signed integer matric multiplication
//
//                            --               --
//                            |  k0  k1  ..  k7 |
//                            |  .   .   .   .  |
//    --              --      |  .   .   .   .  |     --               --         --                 --
//    | i0 - - - - - - |  x   |  .   .   .   .  |  +  |  b0  b1  ..  b7 |     =   | i0k0 i0k1 .. i0k7 |
//    --              --      |  .   .   .   .  |     --               --         --                 --
//                            |  .   .   .   .  |
//                            |  .   .   .   .  |
//                            --               --
//      input 1 x p              kernel p x 8            biases 1 x 8                 output 1 x 8           p = kernel size
//
//
// optimised for Cortex-A72 pipeline 14 cycle per loop (1*8*8 dot product) but the bottleneck is memory bandwidth
//
// input:
//         x0   arg0   biases start address      {b0, b1, b2, b3, b4, b5, b6, b7}
//         x1   arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg2   kernel data start address {k0[0-1],k1[0-1],k2[0-1],k3[0-1],k4[0-1],k5[0-1],k6[0-1],k7[0-1],k0[2-3],k1[2-3],...}
//         x3   arg3   kernel size need to be even number
//         x4   arg4   output data save address  {ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7}
//         x5   arg5   scale address
//
// output: no
//
// v0  8byte data of input {i0[1-0], i0[1-0], i0[1-0], i0[1-0]} 
// v1  8byte data of input {i0[3-2], i0[3-2], i0[3-2], i0[3-2]}
// v2  8byte data of input {i0[5-4], i0[5-4], i0[5-4], i0[5-4]} 
// v4  8byte data of input {i0[7-6], i0[7-6], i0[7-6], i0[7-6]}
// v5 ~ v15 not used
// v16 8byte data of kernel{k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// v17 8byte data of kernel{k7[1-0], k6[1-0], k5[1-0], k4[1-0]}
// v18 8byte data of kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2]}
// v19 8byte data of kernel{k7[3-2], k6[3-2], k5[3-2], k4[3-2]}
// v20 8byte data of kernel{k3[5-4], k2[5-4], k1[5-4], k0[5-4]}
// v21 8byte data of kernel{k7[5-4], k6[5-4], k5[5-4], k4[5-4]}
// v22 8byte data of kernel{k3[7-6], k2[7-6], k1[7-6], k0[7-6]}
// v23 8byte data of kernel{k7[7-6], k6[7-6], k5[7-6], k4[7-6]}
// v24 ~ V27 temp register
// V28 ~ V29 not used
// v30 dot product for  {ik3,  ik2,  ik1,  ik0}
// v31 dot product for  {ik7,  ik6,  ik5,  ik4}

        .section .text,"ax"
        .align 5

        .type i8gemv_1x8_a72 STT_FUNC
        .global i8gemv_1x8_a72
        .hidden i8gemv_1x8_a72
i8gemv_1x8_a72:
// initial
	prfm	pldl1keep, [x2, 0x240]
	cmp     x3,  0x8
	and     x10, x3, 0x3
	movi	d30, 0
	movi	d31, 0

	b.lt	last4
	lsr	x9, x3, 0x3		// x9 = kernel_size / 8
	and	x3, x3, 0x7
	
// main loop     each loop generate dot prodcut for 1x8x8 data
loop8:  
	ld4r	{v0.4h,v1.4h,v2.4h,v3.4h},[x1]// d0={4{i[1-0]}} d1={4{i[3-2]}} d2={4{i[5-4]}} d3={4{i[7-6]}}
	add	x1, x1, 0x8
	ldp	d16,d17, [x2]
	ldp	d18,d19, [x2, 0x10]
	subs	x9, x9, 1		// loop counter
	ldp	d20,d21, [x2, 0x20]
	ldp	d22,d23, [x2, 0x30]

	smull	v24.8h, v16.8b, v0.8b	// ik[3-0][1-0]
	smlal	v24.8h, v18.8b, v1.8b   // ik[3-0][3-2]
	prfm	pldl1keep, [x2, 0x200]
	smull	v25.8h, v17.8b, v0.8b	// ik[7-4][1-0]
	add	x2, x2, 0x40
	smlal	v25.8h, v19.8b, v1.8b	// ik[7-4][3-2]
	smull	v26.8h, v20.8b, v2.8b	// ik[3-0][5-4]
	smlal	v26.8h, v22.8b, v3.8b   // ik[3-0][7-6]
	sadalp	v30.4s, v24.8h
	prfm	pldl1keep, [x1, 0x50]
	smull	v27.8h, v21.8b, v2.8b	// ik[7-4][5-4]
	smlal	v27.8h, v23.8b, v3.8b	// ik[7-4][7-6]
	sadalp	v31.4s, v25.8h
	sadalp	v30.4s, v26.8h
	sadalp	v31.4s, v27.8h

	b.ne	loop8

last4:
	cmp	x3, 0x4
	b.lt	last2

	ld2r    {v0.4h, v1.4h},[x1]	// d0 =({4{i0[1-0]}} d1 =({4{i0[3-2]}}
	add	x1, x1, 0x4
	ldp	d16,d17, [x2]
	ldp	d18,d19, [x2, 0x10]
	smull	v24.8h, v16.8b, v0.8b	// ik[3-0][1-0]
	smlal	v24.8h, v18.8b, v1.8b   // ik[3-0][3-2]
	smull	v25.8h, v17.8b, v0.8b	// ik[7-4][1-0]
	smlal	v25.8h, v19.8b, v1.8b	// ik[7-4][3-2]
	add	x2, x2, 0x20
	sadalp	v30.4s, v24.8h
	sadalp	v31.4s, v25.8h

last2:
	cbz	x10, int2fp

	ld1r	{v0.4h}, [x1]		// d0 =({4{i0[1-0]}}
	ldp	d16,d17, [x2]
	smull	v24.8h, v16.8b, v0.8b	// ik[3-0][1-0]
	smull	v25.8h, v17.8b, v0.8b	// ik[7-4][1-0]
	sadalp	v30.4s, v24.8h
	sadalp	v31.4s, v25.8h

int2fp:
	// convert result to sp and multiply with scale
        ldr     s0, [x5]
        scvtf   v30.4s, v30.4s
        scvtf   v31.4s, v31.4s
	fmul    v30.4s, v30.4s, v0.s[0]
	fmul    v31.4s, v31.4s, v0.s[0]

	// load and add biases
	cbz	x0, save_result
	ldp	q0, q1, [x0]
	fadd	v30.4s, v30.4s, v0.4s
	fadd	v31.4s, v31.4s, v1.4s

save_result:
	stp	q30, q31, [x4]

	ret

        .end
