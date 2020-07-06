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
// 4*4 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                   --
//    | i0 - - - - - - |      |  k0  k1  k2  k3 |     |  b0  b1  b2  b3 |         | i0k0 i0k1 i0k2 i0k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i1k0 i1k1 i1k2 i1k3 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                     |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i2k0 i2k1 i2k2 i2k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i3k0 i3k1 i3k2 i3k3 |
//    --              --      --               --     --               --         --                   --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 4         p = kernel size
//
//
// optimised for Cortex-A17 pipeline 33 cycle per loop (4*4*4 dot product)
//
// input:
//         r0     arg0  input  address {i[0-3][0],i[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         r1     arg1  kernel address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],k[0-3][4],...}
//         r2     arg2  kernel size
//         r3     arg3  output address output                    : {i0k0  i1k0  i2k0  i3k0}
//                                     output + weight size      : {i0k1  i1k1  i2k1  i3k1}
//                                     output + weight size * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                     output + weight size * 3  : {i0k3  i1k3  i2k3  i3k3}
//         sp+0x4 arg4  weight size 
//
// output: no
//
// d0  dot product for {i1k0, i0k0}
// d1  dot product for {i3k0, i2k0}
// d2  dot product for {i1k1, i0k1}
// d3  dot product for {i3k1, i2k1}
// d4  dot product for {i1k2, i0k2}
// d5  dot product for {i3k2, i2k2}
// d6  dot product for {i1k3, i0k3}
// d7  dot product for {i3k3, i2k3}
// d8  2S input data   { i1 |  i0 }
// d9  2S input data   { i3 |  i2 }
// d10 2S input data   { i1 |  i0 }
// d11 2S input data   { i3 |  i2 }
// d12 2s kernel data  { k1 |  k0 }
// d13 2s kernel data  { k3 |  k2 }
// d14-d15 not used

	.section .text, "ax"
	.align 5

	.type sgemm_4x4_deconv_a17 STT_FUNC
	.global sgemm_4x4_deconv_a17
	.hidden sgemm_4x4_deconv_a17

sgemm_4x4_deconv_a17:
	push		{r4, lr}
	vpush		{d8 - d13}
	vmov.i64	q0, #0x0
	vmov.i64	q1, #0x0
	vmov.i64	q2, #0x0
	vmov.i64	q3, #0x0

	cmp		r2, #0x4
	blt		loop4_end
	lsr		r4, r2, #0x2		// kernel_size / 4

// main loop    each loop generate dot prodcut for 4x4x4SFP
loop4:
	vldr		d12, [r1]		// k0[1-0]
	vldr		d8,  [r0]		// i0[1-0]
	vldr		d9,  [r0, #0x8]		// i1[1-0]
	subs		r4, r4, #0x1
	vmla.f32	q0, q4, d12[0]
	vldr		d13, [r1, #0x8]		// k1[1-0]
	vmla.f32	q1, q4, d12[1]
	vldr		d10, [r0, #0x10]	// i2[1-0]
	vmla.f32	q2, q4, d13[0]
	vldr		d11, [r0, #0x18]	// i3[1-0]
	vldr		d12, [r1, #0x10]	// k2[1-0]
	vmla.f32	q3, q4, d13[1]
	vldr		d13, [r1, #0x18]	// k3[1-0]
	vmla.f32	q0, q5, d12[0]
	vldr		d8,  [r0, #0x20]	// i0[3-2]
	vmla.f32	q1, q5, d12[1]
	vldr		d9,  [r0, #0x28]	// i1[3-2]
	vmla.f32	q2, q5, d13[0]
	vldr		d12, [r1, #0x20]	// k0[3-2]
	vmla.f32	q3, q5, d13[1]
	vldr		d13, [r1, #0x28]	// k1[3-2]
	vmla.f32	q0, q4, d12[0]
	vmla.f32	q1, q4, d12[1]
	vldr		d10, [r0, #0x30]	// i2[3-2]
	vmla.f32	q2, q4, d13[0]
	vldr		d11, [r0, #0x38]	// i3[3-2]
	vmla.f32	q3, q4, d13[1]
	vldr		d12, [r1, #0x30]	// k2[3-2]
	vmla.f32	q0, q5, d12[0]
	vldr		d13, [r1, #0x38]	// k3[3-2]
	vmla.f32	q1, q5, d12[1]
	pld		[r1, #0x180]
	add		r1, r1, #0x40
	vmla.f32	q2, q5, d13[0]
	pld		[r0, #0x180]
	add		r0, r0, #0x40
	vmla.f32	q3, q5, d13[1]
	bne		loop4

loop4_end:
	ands		r2, r2, #0x3
	ldr		r4,[sp, #0x38]	// r4 = weight size
	beq		save_result


loop1:
	vldm		r0!, {d8  -  d9}	// i[3-0]0
	vldm		r1!, {d12 - d13}	// k[3-0]0
	subs		r2, r2, #0x1
	vmla.f32	q0, q4, d12[0]
	vmla.f32	q1, q4, d12[1]
	vmla.f32	q2, q4, d13[0]
	vmla.f32	q3, q4, d13[1]
	bne		loop1

save_result:
	mov		r0, r3
	add		r1, r0, r4, LSL #2
	add		r2, r0, r4, LSL #3
	add		r3, r1, r4, LSL #3


	vst4.32		{d0[0],d2[0],d4[0],d6[0]}, [r0]
	vst4.32		{d0[1],d2[1],d4[1],d6[1]}, [r1]
	vst4.32		{d1[0],d3[0],d5[0],d7[0]}, [r2]
	vst4.32		{d1[1],d3[1],d5[1],d7[1]}, [r3]

	vpop		{d8 - d13}
	pop		{r4, pc}

	.end
