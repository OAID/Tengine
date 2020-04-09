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
 * Copyright (c) 2017, Open AI Lab
 * Author: xiaowei@openailab.com
 */

//
// 1*8 single precise floating point matric multiplication
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
// optimised for Cortex-A72 pipeline 13 cycle per loop (1*8*4 dot product) 
// the bottleneck is memory bandwidth
//
// input:
//         x0   arg0   biases start address      {b0, b1, b2, b3, b4, b5, b6, b7}}
//         x1   arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg2   kernel data start address {k00, k10, k20, k30, k40, k50, k60, k70, k80, k01, k11, k21, k31, ...}
//         x3   arg3   kernel size
//         x4   arg4   output data save address  {ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7, ik8}
//
// output: no
//
// v0  4S data of input {i3    i2    i1    i0 }
// v1~v7  not used
// v16 4S kernal data0  {k30 | k20 | k10 | k00}
// v17 4S kernal data4  {k70 | k60 | k50 | k40}
// v18 4S kernal data1  {k31 | k21 | k11 | k01}
// v19 4S kernal data5  {k71 | k61 | k51 | k41}
// v20 4S kernal data2  {k32 | k22 | k12 | k02}
// v21 4S kernal data6  {k72 | k62 | k52 | k42}
// v22 4S kernal data3  {k33 | k23 | k13 | k03}
// v23 4S kernal data7  {k73 | k63 | k53 | k43}
// v24-v29 not used
// v30 dot product for  {ik3,  ik2,  ik1,  ik0}
// v31 dot product for  {ik7,  ik6,  ik5,  ik4}

        .section .text,"ax"
        .align 5

        .type sgemv_1x8_a72 STT_FUNC
        .global sgemv_1x8_a72
        .hidden sgemv_1x8_a72
sgemv_1x8_a72:
// initial
	cmp	x3, 0x4
	prfm	pldl1keep, [x1, 0x40] 
	prfm	pldl1keep, [x2, 0x200] 
	prfm	pldl1keep, [x2, 0x240] 
	movi	d30, 0
	movi	d31, 0
	cbz	x0,  start_convolution
        ldp	q30, q31, [x0]  

start_convolution:
	and	x5, x3, 0x3
	b.lt	loop1
	lsr	x6, x3, 0x2


// main loop     each loop generate dot prodcut for 1x8x4SFP
loop4:
	ldr	q0,  [x1]			// q0  = i[3-0]
	ldp     q16, q17, [x2]			// q16 = k[3-0][0]  q17 = k[7-4][0]
	ldp     q18, q19, [x2, 0x20]		// q18 = k[3-0][1]  q19 = k[7-4][1]
	prfm	pldl1keep, [x1, 0x80] 
	add	x1,  x1,  0x10
	subs	x6, x6, 0x1
	
	fmla	v30.4s, v16.4s, v0.s[0]		// ik[3-0][0]
	fmla	v31.4s, v17.4s, v0.s[0]		// ik[7-4][0]
	ldp     q20, q21, [x2, 0x40]		// q20 = k[3-0][2]  q21 = k[7-4][2]
	fmla	v30.4s, v18.4s, v0.s[1]		// ik[3-0][1]
	prfm	pldl1keep, [x2, 0x400] 
	fmla	v31.4s, v19.4s, v0.s[1]		// ik[7-4][1]
	ldp     q22, q23, [x2, 0x60]		// q22 = k[3-0][3]  q23 = k[7-4][3]
	fmla	v30.4s, v20.4s, v0.s[2]		// ik[3-0][2]
	prfm	pldl1keep, [x2, 0x440] 
	add	x2,  x2,  0x80
	fmla	v31.4s, v21.4s, v0.s[2]		// ik[7-4][2]
	fmla	v30.4s, v22.4s, v0.s[3]		// ik[3-0][3]
	fmla	v31.4s, v23.4s, v0.s[3]		// ik[7-4][3]

	b.ne	loop4

	cbz	x5, save_result

loop1:
	ldr	s0, [x1], 0x4
	ldp	q16, q17, [x2], 0x20
	subs	x5, x5, 0x1

	fmla	v30.4s, v16.4s, v0.s[0]
	fmla	v31.4s, v17.4s, v0.s[0]

	b.ne	loop1
	
save_result:
	stp	q30, q31, [x4]

	ret
 
        .end

