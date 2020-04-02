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
// 1*2 single precise floating point matric multiplication
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
// optimised for Cortex-A72 pipeline 9 cycle per loop (1*2*4 dot product) 
// the bottleneck is memory bandwidth
//
// input:
//         x0   arg0   biases start address      {b0, b1 }
//         x1   arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         x2   arg2   kernel data start address {k00, k10, k01, k11, k02, k12, ...}
//         x3   arg3   kernel size
//         x4   arg4   output data save address  {ik0, ik1}
//
// output: no
//
// v0  4S data of input {i3 i2 i1 i0 }
// v1~v7  not used
// v16 2S kernal data0  {k10 | k00}
// v17    not used 
// v18 2S kernal data1  {k11 | k01}
// v19    not used 
// v20 2S kernal data2  {k12 | k02}
// v21    not used 
// v22 2S kernal data3  {k13 | k03}
// v23    not used 
// v24-29 not used
// v30 dot product for  {ik1,  ik0}
// v31 dot product for  {ik1,  ik0}

        .section .text,"ax"
        .align 5

        .type sgemv_1x2_a72 STT_FUNC
        .global sgemv_1x2_a72
        .hidden sgemv_1x2_a72
sgemv_1x2_a72:
// initial
	movi	d30, 0
	prfm	pldl1keep, [x1, 0x80] 
	cmp	x3, 0x4
	prfm	pldl1keep, [x2, 0x100] 
	prfm	pldl1keep, [x2, 0x140] 
	cbz	x0,  start_convolution
        ldr	d30, [x0]  

start_convolution:
	and	x10,x3, 0x3
	b.lt	loop4_end
	movi	d31, 0
	lsr	x9, x3, 0x2


// main loop     each loop generate dot prodcut for 1x2x4SFP
loop4:
	ldr	q0,  [x1]			// q0  = i[3-0]
	ldp     d16, d18, [x2]			// d16 = k[1-0][0]  d18 = k[1-0][1]
	ldp     d20, d22, [x2, 0x10]		// d20 = k[1-0][2]  d22 = k[1-0][3]
	prfm	pldl1keep, [x1, 0x100] 
	add	x1,  x1,  0x10
	
	fmla	v30.2s, v16.2s, v0.s[0]		// ik[1-0][0]
	subs	x9, x9, 0x1
	fmla	v31.2s, v18.2s, v0.s[1]		// ik[1-0][1]
	prfm	pldl1keep, [x2, 0x200] 
	add	x2,  x2,  0x20
	fmla	v30.2s, v20.2s, v0.s[2]		// ik[1-0][2]
	fmla	v31.2s, v22.2s, v0.s[3]		// ik[1-0][3]

	b.ne	loop4
	fadd	v30.2s, v30.2s, v31.2s

loop4_end:
	cbz	x10, save_result

loop1:
	ldr	s0, [x1], 0x4
	ldr	d16,[x2], 0x8
	subs	x10,x10, 0x1

	fmla	v30.2s, v16.2s, v0.s[0]

	b.ne	loop1
	
save_result:
	str	d30, [x4]

	ret

        .end

