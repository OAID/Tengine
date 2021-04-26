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
// 1*8 INT8 matric multiplication
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
// optimised for Cortex-A7 pipeline 18 cycle per loop (1*8*8 dot product)
//
// input:
//         r0     arg0  biases address {b0, b1, b2, b3, b4, b5, b6, b7}  nullptr means no biases
//         r1     arg1  input  address {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         r2     arg2  kernel address {k[0-7][0-1],k[0-7][2-3],k[0-7][4-5],k[0-7][6-7],...}
//         r3     arg3  kernel size need to be even number
//         sp     arg4  output address {ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7, ik8}
//         sp+0x4 arg5  scale address
//
// output: no
//
// d0  8byte input {i0[1-0], i0[1-0], i0[1-0], i0[1-0]}
// d1  8byte input {i0[3-2], i0[3-2], i0[3-2], i0[3-2]}
// d2  8byte kernel{k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// d3  8byte kernel{k7[1-0], k6[1-0], k5[1-0], k4[1-0]}
// d4  8byte kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2]} 
// d5  8byte kernel{k7[3-2], k6[3-2], k5[3-2], k4[3-2]} 
// d6-11 temp reulst 
// q6  dot product {ik3,  ik2,  ik1,  ik0}
// q7  dot product {ik7,  ik6,  ik5,  ik4}

        .section .text,"ax"
        .align 5

        .type i8gemv_1x8_a17_chan STT_FUNC
        .global i8gemv_1x8_a17_chan
        .hidden i8gemv_1x8_a17_chan
i8gemv_1x8_a17_chan:
	cmp		r3, #0x8
	vpush		{d8 - d15}
	vmov.i64	q6, #0x0
	vmov.i64	q7, #0x0

	blt		loop8_end
	lsr		r12,r3, #0x3	// kernel_size / 8

// main loop    each loop generate 1x8x8 dot product
loop8:
	vldm		r1!, {d6}
	vldr		d2, [r2]
	vldr		d3, [r2, #0x8]
	vdup.16		d0, d6[0]
	vdup.16		d1, d6[1]
	vldr		d4, [r2, #0x10]
	vldr		d5, [r2, #0x18]
	vmull.s8	q4, d0, d2
	vmull.s8	q5, d0, d3
	vldr		d2, [r2, #0x20]
	vldr		d3, [r2, #0x28]
	vmlal.s8	q4, d1, d4
	vmlal.s8	q5, d1, d5
	vpadal.s16	q6, q4
	vpadal.s16	q7, q5
        vdup.16         d0, d6[2]
        vdup.16         d1, d6[3]
	vldr		d4, [r2, #0x30]
	vldr		d5, [r2, #0x38]
        vmull.s8        q4, d0, d2
        vmull.s8        q5, d0, d3
	add		r2, r2, #0x40
        vmlal.s8        q4, d1, d4
        vmlal.s8        q5, d1, d5
        vpadal.s16      q6, q4
        vpadal.s16      q7, q5
	pld		[r1, #0x40]
	subs		r12, r12, #0x1
	pld		[r2, #0x140]
	bne		loop8

loop8_end:
	tst		r3, #0x4
	beq		last2
	
// last 6 data
	vldm		r1!, {s12}
	vldm		r2!, {d2,d3}
	vdup.16		d0, d6[0]
	vdup.16		d1, d6[1]
	vldm		r2!, {d4-d5}
	vmull.s8	q4, d0, d2
	vmull.s8	q5, d0, d3
	vmlal.s8	q4, d1, d4
	vmlal.s8	q5, d1, d5
	vpadal.s16	q6, q4
	vpadal.s16	q7, q5

last2:
// last 2 data
	ands		r3, r3, #0x3
	beq		int2fp

	ldrh		r1, [r1]
	vldm		r2, {d2-d3}
	vdup.16		d0, r1
	vmull.s8	q4, d0, d2
	vmull.s8	q5, d0, d3
	vpadal.s16	q6, q4
	vpadal.s16	q7, q5

int2fp:
	// convert result to sp and multiply with scale
        ldr     	r2, [sp, #0x44]			// r2 = scale address
        vcvt.f32.s32	q6, q6
        vldm		r2, {d0-d3}
        vcvt.f32.s32	q7, q7
        vmul.f32	q6, q6, q0
        vmul.f32	q7, q7, q1

	// load and add biases
	teq		r0, #0x0
	beq		save_result
        vldm		r0, {d0-d3}
        vadd.f32	q6, q6, q0	
        vadd.f32	q7, q7, q1	


save_result:
	ldr		r0, [sp, #0x40]			// r0 = output address 
	vstm		r0, {d12-d15}

	vpop		{d8 - d15}
	bx		lr

        .end
