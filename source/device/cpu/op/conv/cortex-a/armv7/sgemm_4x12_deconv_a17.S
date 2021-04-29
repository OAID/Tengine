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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xiaowei@openailab.com
 */

//
// 4*12 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                   --
//    | i0 - - - - - - |      |  k0  k1  .   kb |     |  b0  b1  .   bb |         | i0k0 i0k1 ..   i0kb |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bb |         | i1k0 i1k1 ..   i1kb |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                     |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bb |         | i2k0 i2k1 ..   i2kb |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bb |         | i3k0 i3k1 ..   i3kb |
//    --              --      --               --     --               --         --                   --
//      input 4 x p             kernel p x 12             biases 4 x 12                 output 4 x 12         p = kernel size
//
//
// optimised for Cortex-A17 pipeline ?? cycle per loop (4*12*4 dot product)
//
// input:
//         r0     arg0  input  address {i[0-3][0],i[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         r1     arg1  kernel address {k[0-b][0],k[0-b][1],k[0-b][2],k[0-b][3],k[0-b][4],...}
//         r2     arg2  kernel size
//         r3     arg3  output address output                    : {i0k0~k11}
//                                     output + weight size      : {i0k1~k11}
//                                     output + weight size * 2  : {i0k1~k11}
//                                     output + weight size * 3  : {i0k1~k11}
//         sp     arg4  weight size 
//
// output: no
//
// q0  4S input data   { i3   i2  i1   i0 }
// q1  4s kernel data  { k3   k2  k1   k0 }
// q2  4s kernel data  { k7   k6  k5   k4 }
// q3  4s kernel data  { kb   ka  k9   k8 }
// q4  dot product for {i3k0, i2k0, i1k0, i0k0}
// q5  dot product for {i3k1, i2k1, i1k1, i0k1}
// q6  dot product for {i3k2, i2k2, i1k2, i0k2}
// q7  dot product for {i3k3, i2k3, i1k3, i0k3}
// q8  dot product for {i3k4, i2k4, i1k4, i0k4}
// q9  dot product for {i3k5, i2k5, i1k5, i0k5}
// q10 dot product for {i3k6, i2k6, i1k6, i0k6}
// q11 dot product for {i3k7, i2k7, i1k7, i0k7}
// q12 dot product for {i3k8, i2k8, i1k8, i0k8}
// q13 dot product for {i3k9, i2k9, i1k9, i0k9}
// q14 dot product for {i3ka, i2ka, i1ka, i0ka}
// q15 dot product for {i3kb, i2kb, i1kb, i0kb}


	.section .text, "ax"
	.align 5

	.type sgemm_4x12_deconv_a17 STT_FUNC
	.global sgemm_4x12_deconv_a17
	.hidden sgemm_4x12_deconv_a17

sgemm_4x12_deconv_a17:
	pld		[r0,#0x80]
	push		{r4, lr}
	vpush		{d8-d15}

	vmov.i64	q4,  #0x0
	vmov.i64	q5,  #0x0
	vmov.i64	q6,  #0x0
	vmov.i64	q7,  #0x0
	vmov.i64	q8,  #0x0
	vmov.i64	q9,  #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	vmov.i64	q12, #0x0
	vmov.i64	q13, #0x0
	vmov.i64	q14, #0x0
	vmov.i64	q15, #0x0

	cmp		r2, #0x4
	blt		loop4_end
	lsr		r4, r2, #0x2		// kernel_size / 4

// main loop    each loop generate dot prodcut for 4x12x4SFP
loop4:
	vldm		r0!,{d0-d1}		// i[3-0][0]
	vldm		r1,{d2-d7}		// k[11-0][0]
	subs		r4, r4, #1
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vldr		d2,[r1,#0x30]
	vldr		d3,[r1,#0x38]
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]
	vmla.f32	q12,q0, d6[0]
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
	vldm		r0!,{d0-d1}		// i[3-0][0]
	vldr		d4,[r1,#0x40]
	vldr		d5,[r1,#0x48]
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vldr		d6,[r1,#0x50]
	vldr		d7,[r1,#0x58]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]
	vldr		d2,[r1,#0x60]
	vldr		d3,[r1,#0x68]
	vmla.f32	q12,q0, d6[0]
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
	vldm		r0!,{d0-d1}		// i[3-0][0]
	vldr		d4,[r1,#0x70]
	vldr		d5,[r1,#0x78]
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vldr		d6,[r1,#0x80]
	vldr		d7,[r1,#0x88]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]
	vldr		d2,[r1,#0x90]
	vldr		d3,[r1,#0x98]
	vmla.f32	q12,q0, d6[0]
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
	vldm		r0!,{d0-d1}		// i[3-0][0]
	vldr		d4,[r1,#0xa0]
	vldr		d5,[r1,#0xa8]
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	pld		[r0,#0x140]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vldr		d6,[r1,#0xb0]
	vldr		d7,[r1,#0xb8]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	pld		[r1,#0x380]
	vmla.f32	q10,q0, d5[0]
	pld		[r1,#0x3c0]
	vmla.f32	q11,q0, d5[1]
	pld		[r1,#0x400]
	vmla.f32	q12,q0, d6[0]
	add		r1, r1, #0xc0
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
	bne		loop4

loop4_end:
	ands		r2, r2, #0x3
	ldr		r4, [sp, #0x48]	// r4 = weight_size
	lsl		r4, r4, #2
	beq		save_result

loop1:
	vldm		r0!,{d0-d1}		// i[3-0][0]
	vldm		r1!,{d2-d7}		// k[11-0][0]
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]
	vmla.f32	q12,q0, d6[0]
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
	subs		r2, r2, #0x1
	bne		loop1

save_result:
	// r0, r1, r2, r3 as base register   r4 = weight size
	mov		r0, r3
	add		r1, r0, r4
	add		r2, r0, r4, LSL #1
	add		r3, r1, r4, LSL #1

	vst4.32		{d8[0], d10[0],d12[0],d14[0]}, [r0]!
	vst4.32		{d8[1], d10[1],d12[1],d14[1]}, [r1]!
	vst4.32		{d9[0], d11[0],d13[0],d15[0]}, [r2]!
	vst4.32		{d9[1], d11[1],d13[1],d15[1]}, [r3]!
	vst4.32		{d16[0],d18[0],d20[0],d22[0]}, [r0]!
	vst4.32		{d16[1],d18[1],d20[1],d22[1]}, [r1]!
	vst4.32		{d17[0],d19[0],d21[0],d23[0]}, [r2]!
	vst4.32		{d17[1],d19[1],d21[1],d23[1]}, [r3]!
	vst4.32		{d24[0],d26[0],d28[0],d30[0]}, [r0]
	vst4.32		{d24[1],d26[1],d28[1],d30[1]}, [r1]
	vst4.32		{d25[0],d27[0],d29[0],d31[0]}, [r2]
	vst4.32		{d25[1],d27[1],d29[1],d31[1]}, [r3]

	vpop		{d8-d15}
	pop		{r4,pc}

	.end
