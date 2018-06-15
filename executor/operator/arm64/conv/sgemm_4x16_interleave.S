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
// 4*16 single precise floating point matric multiplication
//
//    --              --      --               --     --                --         --                  --
//    | i0 - - - - - - |      |  k0  k1  ..  kf |     |  t00 t01  .. t0f |         | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |     |                  |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  t10 t11 .   t1f |         | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |  +  |                  |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  t20 t21 .   t2f |         | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |     |                  |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  t30 t31 .   t3f |         | i3k0 i3k1 .. i3kf |
//    --              --      --               --     --                --         --                  --
//      input 4 x p             kernel p x 16            biases 4 x 16                 output 4 x 16           p = kernel size
//
//
// optimised for Cortex-A72 pipeline  66 cycle per loop (4*16*4 dot product)
//
// input: 
//         x0 arg0  have biases flag
//         x1 arg1  biases start address {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3],i[0-3]k[4]..} 
//         x2 arg2  input  start address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x3 arg3  kernel start address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],...}
//         x4 arg4  output save  address {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3],i[0-3]k[4]..}
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
// v0~v1  4S data of input0   {i3   i2   i1   i0}
// v2-v3 not used
// v4  4S kernal data      {k3 | k2 | k1 | k0}
// v5  4S kernal data      {k7 | k6 | k5 | k4}
// v6  4S kernal data      {kb | ka | k9 | k8}
// v7  4S kernal data      {kf | ke | kd | kc}
// v8~v15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20 dot product for {i3k4, i2k4, i1k4, i0k4}
// v21 dot product for {i3k5, i2k5, i1k5, i0k5}
// v22 dot product for {i3k6, i2k6, i1k6, i0k6}
// v23 dot product for {i3k7, i2k7, i1k7, i0k7}
// v24 dot product for {i3k8, i2k8, i1k8, i0k8}
// v25 dot product for {i3k9, i2k9, i1k9, i0k9}
// v26 dot product for {i3ka, i2ka, i1ka, i0ka}
// v27 dot product for {i3kb, i2kb, i1kb, i0kb}
// v28 dot product for {i3kc, i2kc, i1kc, i0kc}
// v29 dot product for {i3kd, i2kd, i1kd, i0kd}
// v30 dot product for {i3ke, i2ke, i1ke, i0ke}
// v31 dot product for {i3kf, i2kf, i1kf, i0kf}

#ifndef INTERLEAVE_FUNC_NAME
#define INTERLEAVE_FUNC_NAME sgemm_4x16_interleave
#endif

        .section .text,"ax"
        .align 5

        .type INTERLEAVE_FUNC_NAME STT_FUNC
        .global INTERLEAVE_FUNC_NAME

INTERLEAVE_FUNC_NAME:
// biases_initial
	cbz	x0, none_biases
        ldp     q16, q17 ,[x1]
        ldp     q18, q19 ,[x1, #0x20]
        ldp     q20, q21 ,[x1, #0x40]
        ldp     q22, q23 ,[x1, #0x60]
        ldp     q24, q25 ,[x1, #0x80]
        ldp     q26, q27 ,[x1, #0xa0]
        ldp     q28, q29 ,[x1, #0xc0]
        ldp     q30, q31 ,[x1, #0xe0]
	b	convolution_start

none_biases:
	movi	d16, #0
	movi	d17, #0
	movi	d18, #0
	movi	d19, #0
	movi	d20, #0
	movi	d21, #0
	movi	d22, #0
	movi	d23, #0
	movi	d24, #0
	movi	d25, #0
	movi	d26, #0
	movi	d27, #0
	movi	d28, #0
	movi	d29, #0
	movi	d30, #0
	movi	d31, #0

convolution_start:
	// compare to 0x4
	cmp	x5, 0x4
	blt	loop4_end
	lsr	x6, x5, 0x2

// main loop     each loop generate dot prodcut for 4x16SFP
loop4:  
	ldr	q0, [x2]			// q0=i[3-0]
	ldp	q4, q5, [x3]			// q4=k[3-0] q5=k[7-4] 
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	ldp	q6, q7, [x3, 0x20]		// q6=k[b-8] q7=k[f-c]
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	ldr	q1, [x2, 0x10]			// q1=i[3-0]
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	ldp	q4, q5, [x3, 0x40]		// q4=k[3-0] q5=k[7-4] 
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x3, 0x60]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v1.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v1.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q0, [x2, 0x20]			// q1=i[3-0]
	fmla	v20.4s, v1.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v1.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v1.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v1.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x3, 0x80]		// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v1.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v1.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v1.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v1.4s,  v6.s[3]		// i[3-0]k[b]
	subs	x6, x6, #0x1
	prfm	pldl1keep, [x2, 0x80]
	fmla	v28.4s, v1.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v1.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v1.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v1.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x3, 0xa0]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q1, [x2, 0x30]			// q1=i[3-0]
	add	x2, x2, #0x40
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x3, 0xc0]		// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	prfm	pldl1keep, [x3, 0x140]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x3, 0xe0]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v1.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v1.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v4.s[3]		// i[3-0]k[3]
	prfm	pldl1keep, [x3, 0x180]
	fmla	v20.4s, v1.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v1.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v1.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v1.4s,  v5.s[3]		// i[3-0]k[7]
	prfm	pldl1keep, [x3, 0x1c0]
	fmla	v24.4s, v1.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v1.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v1.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v1.4s,  v6.s[3]		// i[3-0]k[b]
	prfm	pldl1keep, [x3, 0x200]
	add	x3, x3, #0x100
	fmla	v28.4s, v1.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v1.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v1.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v1.4s,  v7.s[3]		// i[3-0]k[f]
	b.ne	loop4

	and	x5, x5, 0x3

loop4_end:
	cbz	x5, finish

loop1:
        ldr     q0, [x2], 0x10                  // q0=i[3-0]
        ldp     q4, q5, [x3]                    // q4=k[3-0] q5=k[7-4]
        ldp     q6, q7, [x3, 0x20]              // q6=k[b-8] q7=k[f-c]
        subs    x5 ,x5 ,0x1
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]
        add     x3, x3, #0x40

        b.ne    loop1


finish:
// store result
#ifdef CONV_RELU_FUSE
        fmov    s0,wzr
        dup     v1.4s,v0.s[0]
        fmax    v16.4s,v16.4s,v1.4s
        fmax    v17.4s,v17.4s,v1.4s
#endif
        stp     q16, q17 ,[x4]

#ifdef CONV_RELU_FUSE
        fmax    v18.4s,v18.4s,v1.4s
        fmax    v19.4s,v19.4s,v1.4s
#endif
        stp     q18, q19 ,[x4, #0x20]

#ifdef CONV_RELU_FUSE
        fmax    v20.4s,v20.4s,v1.4s
        fmax    v21.4s,v21.4s,v1.4s
#endif
        stp     q20, q21 ,[x4, #0x40]

#ifdef CONV_RELU_FUSE
        fmax    v22.4s,v22.4s,v1.4s
        fmax    v23.4s,v23.4s,v1.4s
#endif
        stp     q22, q23 ,[x4, #0x60]

#ifdef CONV_RELU_FUSE
        fmax    v24.4s,v24.4s,v1.4s
        fmax    v25.4s,v25.4s,v1.4s
#endif
        stp     q24, q25 ,[x4, #0x80]

#ifdef CONV_RELU_FUSE
        fmax    v26.4s,v26.4s,v1.4s
        fmax    v27.4s,v27.4s,v1.4s
#endif
        stp     q26, q27 ,[x4, #0xa0]

#ifdef CONV_RELU_FUSE
        fmax    v28.4s,v28.4s,v1.4s
        fmax    v29.4s,v29.4s,v1.4s
#endif
        stp     q28, q29 ,[x4, #0xc0]


#ifdef CONV_RELU_FUSE
        fmax    v30.4s,v30.4s,v1.4s
        fmax    v31.4s,v31.4s,v1.4s
#endif
        stp     q30, q31 ,[x4, #0xe0]

	ret

// zero data to fill out a few more cache lines so the prefetcher doesn't
// cause uninitialized memory to be read

                .space  256
                .end

