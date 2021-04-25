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
// 1*2 INT8 matric multiplication
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
// optimised for Cortex-A17 pipeline 12 cycle per loop (1*2*16 dot product)
//
// input:
//         r0     arg0  biases address {b0, b1}  nullptr means no biases
//         r1     arg1  input  address {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         r2     arg2  kernel address {k[0-1][0-1],k[0-1][2-3],k[0-1][4-5],k[0-1][6-7],...}
//         r3     arg3  kernel size need to be even number
//         sp     arg4  output address {ik0, ik1}
//         sp+0x4 arg5  scale address
//
// output: no
//
// d0  8byte input {i0[3-2], i0[3-2], i0[1-0], i0[1-0]}
// d1  8byte input {i0[7-6], i0[7-6], i0[5-4], i0[5-4]}
// d2  8byte input {i0[b-a], i0[b-a], i0[9-8], i0[9-8]}
// d3  8byte input {i0[f-e], i0[f-e], i0[d-c], i0[d-c]}
// d4  8byte kernel{k1[3-2], k0[3-2], k1[1-0], k0[1-0]}
// d5  8byte kernel{k1[7-6], k0[7-6], k1[5-4], k0[5-4]}
// d6  8byte kernel{k1[b-a], k0[b-a], k1[9-8], k0[9-8]}
// d7  8byte kernel{k1[f-e], k0[f-e], k1[d-c], k0[d-c]}
// d8-11 temp reulst 
// q6  dot product {ik1,  ik0,  ik1,  ik0}
// q7  dot product {ik1,  ik0,  ik1,  ik0}

        .section .text,"ax"
        .align 5

        .type i8gemv_1x2_a17_chan STT_FUNC
        .global i8gemv_1x2_a17_chan
        .hidden i8gemv_1x2_a17_chan
i8gemv_1x2_a17_chan:
	cmp		r3, #0x10
	vpush		{d8 - d15}
	vmov.i64	q7, #0x0

	blt		last8
	vmov.i64	q6, #0x0
	lsr		r12,r3, #0x4	// kernel_size / 16

// main loop    each loop generate 1x2x16 dot product
loop16:
	vldm		r1!, {d1,d2}
	vmov		d0, d1
	vmov		d3, d2
	vldr		d4, [r2]
	vldr		d5, [r2, #0x8]
	vzip.16		d0, d1
	vzip.16		d2, d3
	vldr		d6, [r2, #0x10]
	vldr		d7, [r2, #0x18]
	add		r2, r2, #0x20
	vmull.s8	q4, d4, d0
	vmull.s8	q5, d5, d1
	vmlal.s8	q4, d6, d2
	vmlal.s8	q5, d7, d3
	vpadal.s16	q6, q4
	vpadal.s16	q7, q5
	pld		[r1, #0xa0]
	subs		r12, r12, #0x1
	pld		[r2, #0x140]
	bne		loop16

	vadd.i32	q7, q7, q6

last8:
	tst		r3, #0x8
	beq		last4

        vldm            r1!, {d1}
        vmov            d0, d1
        vldm            r2!, {d4-d5}
        vzip.16         d0, d1
        vmull.s8        q4, d4, d0
        vmlal.s8        q4, d5, d1
        vpadal.s16      q7, q4

last4:
	tst		r3, #0x4
	beq		last2

        vldm            r1!, {s0}
        vldm            r2!, {d4}
        vmov            s2, s0
        vzip.16         d0, d1
        vmull.s8        q4, d4, d0
        vpadal.s16      q7, q4

last2:
	ands		r3, r3, #0x3
	beq		int2fp

	ldrh		r1, [r1]
	vldr		s2, [r2]
	vdup.16		d0, r1
	vmull.s8	q4, d0, d1
	vpadal.s16	d14,d8

int2fp:
	// convert result to sp and multiply with scale
        ldrd     	r2, r3, [sp, #0x40]		// r2 = output address  r3 = scale address
	vadd.i32	d15, d15, d14
	vldr		d0, [r3]			// scale
        vcvt.f32.s32	d15, d15
        vmul.f32	d15, d15, d0

	// load and add biases
	teq		r0, #0x0
	beq		save_result
        vldr		d0, [r0]
        vadd.f32	d15, d15, d0	

save_result:
	vstr		d15, [r2]

	vpop		{d8 - d15}
	bx		lr

        .end
