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
// 4*4 single precise floating point matric multiplication
//
//    --              --      --               --     --                --         --                 --
//    | i0 - - - - - - |      |  k0  k1  k2  k3 |     |  t00 t01 t02 t03 |         | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |     |                  |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  t10 t11 t12 t13 |         | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |  +  |                  |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  t20 t21 t22 t23 |         | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |     |                  |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  t30 t31 t32 t33 |         | i3k0 i3k1 .. i3kf |
//    --              --      --               --     --                --         --                 --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 4         p = kernel size
//
//
// optimised for Cortex-A72 pipeline 18 cycle per loop (4*4*4 dot product)
//
// input:  
//         x0 arg0  have biases flag
//         x1 arg1  biases start address {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3]} 
//         x2 arg2  input  start address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x3 arg3  kernel start address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],...}
//         x4 arg4  output save  address {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3]}
//         x5 arg5  kernel size
//
// output: no
//
// register definition
// x0        have biases flag
// x1        biases start address
// x2        input start address
// x3        kernel start address
// x4        output start address
// x5        loop time = kernal size 
// x6 ~ x31 not used
//
// v0-3 4S data of input0   {i3   i2   i1   i0}
// v4-7 4S kernal data      {k3   k2   k1   k0}
// v8~v15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20~V31 not used
        .section .text,"ax"
        .align 5

#ifndef INTERLEAVE_FUNC_NAME
#define INTERLEAVE_FUNC_NAME sgemm_4x4_interleave
#endif
        .type INTERLEAVE_FUNC_NAME STT_FUNC
        .global INTERLEAVE_FUNC_NAME

INTERLEAVE_FUNC_NAME:
// initial
	cbz	x0,  non_biases

	ldp	q16, q17, [x1]
	ldp	q18, q19, [x1,0x20]
	b	convoluation_start
	
non_biases:
	movi	d16, #0x0
	movi	d17, #0x0
	movi	d18, #0x0
	movi	d19, #0x0

convoluation_start:
	// compare to 0x4
	cmp	x5, 0x4
	blt	loop4_end
	lsr	x6, x5, 0x2

// main loop     each loop generate dot prodcut for 4x4SFP
loop4:  
	subs	x6 ,x6 ,0x1

	ldr	q0, [x2]			// q0=i[3-0]
	ldp	q4, q5, [x3]			// q4=k[3-0] 
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	ldr	q1, [x2, 0x10]			// q1=i[3-0]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]

	ldp	q2, q3, [x2, 0x20]		// q2=i[3-0] q3=i[3-0]
	fmla	v16.4s, v1.4s,  v5.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v5.s[1]		// i[3-0]k[1]
	ldp	q6, q7, [x3, 0x20]		// q6=k[3-0] q7=q7=k[3-0]
	fmla	v18.4s, v1.4s,  v5.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v5.s[3]		// i[3-0]k[3]

	fmla	v16.4s, v2.4s,  v6.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v2.4s,  v6.s[1]		// i[3-0]k[1]
	prfm	pldl1keep, [x2, 0x140]
	add	x2, x2, #0x40
	fmla	v18.4s, v2.4s,  v6.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v2.4s,  v6.s[3]		// i[3-0]k[3]

	prfm	pldl1keep, [x3, 0x140]
	add	x3, x3, #0x40
	fmla	v16.4s, v3.4s,  v7.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v3.4s,  v7.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v3.4s,  v7.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v3.4s,  v7.s[3]		// i[3-0]k[3]

	b.ne	loop4

	and	x5, x5, 0x3

loop4_end:
	cbz	x5, finish

loop1:
	subs	x5 ,x5 ,0x1
	ldr     q0, [x2], 0x10                  // q0=i[3-0]
        ldr     q4, [x3], 0x10                  // q4=k[3-0]
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[0]k[3-0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[1]k[3-0]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[2]k[3-0]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3]k[3-0]

	b.ne	loop1
finish:
// store result
#ifdef CONV_RELU_FUSE
        fmov s0,wzr
        dup  v1.4s,v0.s[0]
        fmax v16.4s,v16.4s,v1.4s
        fmax v17.4s,v17.4s,v1.4s
#endif
	stp	q16, q17, [x4]

#ifdef CONV_RELU_FUSE
        fmax v18.4s,v18.4s,v1.4s
        fmax v19.4s,v19.4s,v1.4s
#endif
	stp	q18, q19, [x4,0x20]

	ret

// zero data to fill out a few more cache lines so the prefetcher doesn't
// cause uninitialized memory to be read

                .space  256
                .end

