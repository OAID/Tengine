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
// 1*2 half precise floating point matric multiplication
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
// optimised for Cortex-A53 pipeline 6 cycle per loop (1*2*4 dot product) 
//
// input:
//         x0   arg1   biases start address      {b0, b1}
//         x1   arg2   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg3   kernel data start address {k00, k10, k01, k11, k02, k12, ...}
//         x3   arg5   kernel size
//         x4   arg4   output data save address  {ik0, ik1}
//
// output: no
//
// v0  4h data of input {i3  i2  i1  i0 }
// v1  8h data of input {i3  i3  i2  i2  i1  i1  i0  i0 }
// v1~v7  not used
// v16 8h kernal data0  {k13 k03 k12 k02 k11 k01 k10 k00}
// v18-v30 not used
// v31 2h dot product   {ik1 ik0 ik1 ik0 ik1 ik0 ik1 ik0}

        .section .text,"ax"
        .align 5

        .type hgemv_1x2_a55 STT_FUNC
        .global hgemv_1x2_a55
        .hidden hgemv_1x2_a55
hgemv_1x2_a55:
// initial
	cmp	x3, 0x4
	movi	d31, 0x0
	cbz	x0,  start_convolution
        ldr	s31, [x0]  

start_convolution:
	and	x10,x3, 0x3
	b.lt	loop4_end
	lsr	x9, x3, 0x2


// main loop     each loop generate dot prodcut for 1x2x4 HP
loop4:
	ldr	d0,  [x1], 0x8			// q0  = i[3-0]
	ldr	q16, [x2], 0x10
	zip1	v1.8h,v0.8h,v0.8h		
	subs	x9, x9, 0x1
	fmla	v31.8h,v16.8h, v1.8h		// ik[1-0][0]
	b.ne	loop4

loop4_end:
	mov	v30.d[0], v31.d[1]
	fadd	v29.4h, v31.4h, v30.4h
	mov	v28.s[0], v29.s[1]
	fadd	v31.4h, v28.4h, v29.4h

	cbz	x10, save_result

loop1:
	ldr	h0, [x1], 0x2
	ldr	s16,[x2], 0x4
	subs	x10,x10, 0x1
	fmla	v31.4h, v16.4h, v0.h[0]
	b.ne	loop1

	
save_result:
	str	s31, [x4]

	ret


        .end

