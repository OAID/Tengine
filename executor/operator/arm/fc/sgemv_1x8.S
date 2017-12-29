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
//    --              --      |  .   .   .   .  |     --               --                              
//    | i0 - - - - - - |  x   |  .   .   .   .  |  +  |  b0  b1  ..  b7 |     =   | i0k0 i0k1 .. i0k7 |
//    --              --      |  .   .   .   .  |     --               --         --                 --     
//                            |  .   .   .   .  |                                                      
//                            |  .   .   .   .  |                                                      
//                            --               --                                       
//      input 1 x p              kernel p x 8            biases 1 x 8                 output 1 x 8           p = kernel size
//
//
// optimised for Cortex-A72 pipeline  15 cycle per loop (1*8*4 dot product) 
// the bottleneck is memory bandwidth
//
// input:  
//         x0          arg0   biases start address
//         x1          arg1   input0 data start address
//         x2          arg2   kernel01 data start address {k00,k10,k01,k11,k02,k12,...}
//         x3          arg3   kernel23 data start address {k20,k30,k21,k31,k22,k32,...}
//         x4          arg4   kernel45 data start address {k40,k50,k41,k51,k42,k52,...}
//         x5          arg5   kernal67 data start address {k60,k70,k61,k71,k62,k72,...} 
//         x6          arg6   output0 data save address
//         x7          arg7   output1 data save address
//         x9  sp      arg8   output2 data save address
//         x10 sp+0x8  arg9   output3 data save address
//         x11 sp+0x10 arg10  output4 data save address
//         x12 sp+0x18 arg11  output5 data save address
//         x13 sp+0x20 arg12  output6 data save address
//         x14 sp+0x28 arg13  output7 data save address
//         x15 sp+0x30 arg14  kernel size / 8
//
// output: no
//
// v0  4S data of input {i03   i02   i01   i00}
// v1~v7  not used
// v16 4S kernal data0 {k30 | k20 | k10 | k00}
// v17 4S kernal data1 {k31 | k21 | k11 | k01}
// v18 4S kernal data2 {k32 | k22 | k12 | k02}
// v19 4S kernal data3 {k33 | k23 | k13 | k03}
// v20 4S kernal data4 {k70 | k60 | k50 | k40}
// v21 4S kernal data5 {k71 | k61 | k51 | k41}
// v22 4S kernal data6 {k72 | k62 | k52 | k42}
// v23 4S kernal data7 {k73 | k63 | k53 | k43}
// v24-v29 not used
// v30 dot product for {i0k3, i0k2, i0k1, i0k0}
// v31 dot product for {i0k7, i0k6, i0k5, i0k4}

        .section .text,"ax"
        .align 5

        .type sgemv_1x8 STT_FUNC
        .global sgemv_1x8
sgemv_1x8:
// initial
        // save context
        ldp     x9,  x10, [sp]
        ldp     x11, x12, [sp, 0x10]
        ldp     x13, x14, [sp, 0x20]
        ldr     x15, [sp, 0x30]

        ldp	q30, q31, [x0]  

	cbz	x15, loop_end

// main loop     each loop generate dot prodcut for 1x8x4SFP
loop:
	ldr	q0, [x1], #0x10				// ld i0
	ld4     { v16.d, v17.d, v18.d, v19.d}[0], [x2], #0x20    // ld k0-1
	ld4     { v16.d, v17.d, v18.d, v19.d}[1], [x3], #0x20    // ld k0-1
	ld4     { v20.d, v21.d, v22.d, v23.d}[0], [x4], #0x20    // ld k2-3
	ld4     { v20.d, v21.d, v22.d, v23.d}[1], [x5], #0x20    // ld k2-3

	fmla	v30.4s, v16.4s, v0.s[0]			// i0k[3-0]
	fmla	v30.4s, v17.4s, v0.s[1]			// i0k[3-0]
	subs	x15,x15,0x1
	fmla	v30.4s, v18.4s, v0.s[2]			// i0k[3-0]
	prfm	pldl1keep, [x1, 0x80] 
	fmla	v30.4s, v19.4s, v0.s[3]			// i0k[3-0]
	prfm	pldl1keep, [x2, 0xa0] 

	fmla	v31.4s, v20.4s, v0.s[0]			// i0k[7-4]
	prfm	pldl1keep, [x3, 0x90] 
	fmla	v31.4s, v21.4s, v0.s[1]			// i0k[7-4]
	prfm	pldl1keep, [x4, 0xb0] 
	fmla	v31.4s, v22.4s, v0.s[2]			// i0k[7-4]
	prfm	pldl1keep, [x5, 0xc0] 
	fmla	v31.4s, v23.4s, v0.s[3]			// i0k[7-4]

	b.ne	loop

loop_end:

	
// store result
	st1	{v30.s}[0], [x6]
	st1	{v30.s}[1], [x7]
	st1	{v30.s}[2], [x9]
	st1	{v30.s}[3], [x10]
	st1	{v31.s}[0], [x11]
	st1	{v31.s}[1], [x12]
	st1	{v31.s}[2], [x13]
	st1	{v31.s}[3], [x14]

	ret

// zero data to fill out a few more cache lines so the prefetcher doesn't
// cause uninitialized memory to be read

                .space  256
                .end

