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
// 1*8 half precise floating point matric multiplication
//
//                            --               --
//                            |  k0  k1  ..  k7 |                                                      
//                            |  .   .   .   .  |                                                      
//    --              --      |  .   .   .   .  |     --               --                              
//    | i0 - - - - - - |  x   |  .   .   .   .  |  +  |  b0  b1  ..  b7 |     =   | i0k0 i0k1 .. i0k7 |
//    --              --      |  .   .   .   .  |     --               --         --                 --     
//                            |  .   .   .   .  |                                                      
//                            |  .   .   .   .  |                                                      
//                            --               --                                       
//      input 1 x p              kernel p x 8            biases 1 x 8                 output 1 x 8           p = kernel size
//
//
// optimised for Cortex-A55 pipeline 13 cycle per loop (1*8*4 dot product) 
// the bottleneck is memory bandwidth
//
// input:
//         x0   arg0   biases start address      {b0, b1, b2, b3, b4, b5, b6, b7}
//         x1   arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg2   kernel data start address {k00, k10, k20, k30, k40, k50, k60, k70, k80, k01, k11, k21, k31, ...}
//         x3   arg3   kernel size
//         x4   arg4   output data save address  {ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7, ik8}
//
// output: no
//
// v0  4h data of input {i3    i2    i1    i0 }
// v1~v7  not used
// v16 8h kernal data0  {k70 | k60 | k50 | k40 | k30 | k20 | k10 | k00}
// v17 8h kernal data4  {k71 | k61 | k51 | k41 | k31 | k21 | k11 | k01}
// v18 8h kernal data1  {k72 | k62 | k52 | k42 | k32 | k22 | k12 | k02}
// v19 8h kernal data5  {k73 | k63 | k53 | k43 | k33 | k23 | k13 | k03}
// v20-v30 not used
// v31 8h  dot product  {ik7,  ik6,  ik5,  ik4,  ik3,  ik2,  ik1,  ik0}

        .section .text,"ax"
        .align 5

        .type hgemv_1x8_a55 STT_FUNC
        .global hgemv_1x8_a55
        .hidden hgemv_1x8_a55
hgemv_1x8_a55:
	// initial
	cmp	x3,  0x4
	movi	d31, 0x0
	cbz	x0,  start_convolution
        ldr	q31, [x0]  

start_convolution:
	b.lt	loop1
	lsr	x6, x3, 0x2
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0


// main loop     each loop generate dot prodcut for 1x8x4 HP
loop4:
	ldr	d0,  [x1], 0x8			// q0  = i[3-0]
	ldp     q16, q17, [x2]			// q16 = k[7-0][0]  q17 = k[7-0][1]
	ldp     q18, q19, [x2, 0x20]		// q18 = k[7-0][2]  q19 = k[7-0][3]
	subs	x6, x6, 0x1
	
	fmla	v28.8h, v16.8h, v0.h[0]		// ik[3-0][0]
	fmla	v29.8h, v17.8h, v0.h[1]		// ik[7-4][0]
	fmla	v30.8h, v18.8h, v0.h[2]		// ik[3-0][1]
	add	x2,  x2,  0x40
	fmla	v31.8h, v19.8h, v0.h[3]		// ik[7-4][1]
	b.ne	loop4

	fadd	v29.8h, v29.8h, v28.8h
	and	x3, x3, 0x3
	fadd	v31.8h, v31.8h, v30.8h
	fadd	v31.8h, v31.8h, v29.8h

	cbz	x3, save_result

loop1:
	ldr	h0, [x1], 0x2
	ldr	q16,[x2], 0x10
	subs	x3, x3,   0x1
	fmla	v31.8h, v16.8h, v0.h[0]
	b.ne	loop1
	
save_result:
	str	q31, [x4]

	ret


        .end

