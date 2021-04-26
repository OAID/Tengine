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
// 1*8 single precise floating point matric multiplication
//
//                            --               --
//                            |  k0  k1  ..  k7 |
//                            |  .   .   .   .  |
//    --              --      |  .   .   .   .  |     --               --         --                 --
//    | i0 - - - - - - |  x   |  .   .   .   .  |  +  |  b0  b1  ..  b7 |     =   | i0k0 i0k1 .. i0k7 |
//    --              --      |  .   .   .   .  |     --               --         --                 --
//                            |  .   .   .   .  |
//                            |  .   .   .   .  |
//                            --               --
//      input 1 x p              kernel p x 8            biases 1 x 8                 output 1 x 8           p = kernel size
//
//
// optimised for Cortex-A7 pipeline 59 cycle per loop (1*8*4 dot product)
// the bottleneck is memory bandwidth
//
// input:
//         r0     arg0   biases start address      {b0, b1, b2, b3, b4, b5, b6, b7}   nullptr means no biases
//         r1     arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         r2     arg2   kernel data start address {k00, k10, k20, k30, k40, k50, k60, k70, k80, k01, k11, k21, k31, ...}
//         r3     arg3   kernel size
//         sp     arg4   output data save address  {ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7, ik8}
//
// output: no
//
// register definition
//
// d0  dot product for {ik1, ik0}
// d1  dot product for {ik3, ik2}
// d2  dot product for {ik5, ik4}
// d3  dot product for {ik7, ik6}
// d4  2S kernel data  {k1 | k0 }
// d5  2S kernel data  {k3 | k2 }
// d6  2S kernel data  {k5 | k4 }
// d7  2S kernel data  {k7 | k6 }
// d8  2S kernel data  {k1 | k0 }
// d9  2S kernel data  {k3 | k2 }
// d10 2S kernel data  {k5 | k4 }
// d11 2S kernel data  {k7 | k6 }
// d12 2s input  data  {i1 | i0 }
// d13~d15 not used

	.section .text, "ax"
	.align 5

	.type sgemv_1x8_a7 STT_FUNC
	.global sgemv_1x8_a7
	.hidden sgemv_1x8_a7

sgemv_1x8_a7:
	// context save and load parameters
	vpush		{d8 - d12}

	teq		r0, #0x0		// have_biases flag
	vmov.i64	q0, #0x0
	vmov.i64	q1, #0x0
	vldmne		r0, {d0-d3}

	cmp		r3, #0x4
	blt		loop4_end
	lsr		r0, r3, #0x2		// kernel_size / 4

// main loop    each loop generate dot prodcut for 1x8x4SP
loop4:
	vldr		d4, [r2]		// k10, k00
	vldr		d12,[r1]		// i1, i0
	vldr		d5, [r2, #0x8]		// k30, k20
	vldr		d6, [r2, #0x10]		// k50, k40
	vldr		d7, [r2, #0x18]		// k70, k60
	vldr		d8, [r2, #0x20]		// k11, k01
	vldr		d9, [r2, #0x28]		// k31, k21
	vldr		d10,[r2, #0x30]		// k51, k41
	vldr		d11,[r2, #0x38]		// k71, k61
	vmla.f32	d0, d4, d12[0]
	vmla.f32	d1, d5, d12[0]
	vmla.f32	d2, d6, d12[0]
	vmla.f32	d3, d7, d12[0]
	vmla.f32	d0, d8, d12[1]
	vmla.f32	d1, d9, d12[1]
	vmla.f32	d2, d10,d12[1]
	vmla.f32	d3, d11,d12[1]
	pld		[r1, #0x40]
	pld		[r2, #0x100]
	pld		[r2, #0x140]
	vldr		d12,[r1, #0x8]		// i3, i2
	vldr		d4, [r2, #0x40]		// k12, k02
	vldr		d5, [r2, #0x48]		// k32, k22
	vldr		d6, [r2, #0x50]		// k52, k42
	vldr		d7, [r2, #0x58]		// k72, k62
	vldr		d8, [r2, #0x60]		// k13, k03
	vldr		d9, [r2, #0x68]		// k33, k23
	vldr		d10,[r2, #0x70]		// k53, k43
	vldr		d11,[r2, #0x78]		// k73, k63
	add		r2, r2, #0x80
	add		r1, r1, #0x10
	vmla.f32	d0, d4, d12[0]
	vmla.f32	d1, d5, d12[0]
	vmla.f32	d2, d6, d12[0]
	vmla.f32	d3, d7, d12[0]
	vmla.f32	d0, d8, d12[1]
	vmla.f32	d1, d9, d12[1]
	vmla.f32	d2, d10,d12[1]
	vmla.f32	d3, d11,d12[1]
	subs		r0, r0, #0x1
	bne		loop4

loop4_end:
	ldr		r0, [sp, #0x28]
	ands		r3, #0x3
	beq		save_result

loop1:
	vldr		d4, [r2]		// k10, k00
	vldr		d5, [r2, #0x8]		// k30, k20
	vldr		d6, [r2, #0x10]		// k50, k40
	vldr		d7, [r2, #0x18]		// k70, k60
	vldr		s16,[r1]		// i0
	add		r2, r2, #0x20
	add		r1, r1, #0x4
	vmla.f32	d0, d4, d8[0]
	vmla.f32	d1, d5, d8[0]
	vmla.f32	d2, d6, d8[0]
	vmla.f32	d3, d7, d8[0]
	subs		r3, r3, #0x1
	bne		loop1

save_result:
	vstm 		r0, {d0 - d3}

	vpop		{d8 - d12}
	bx		lr

	.end
