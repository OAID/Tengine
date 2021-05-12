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
//         r0     arg0  biases address    {b0,b1,b2,b3,b4}   nullptr means no biases
//         r1     arg1  input  address {i[0-3][0],i[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         r2     arg2  kernel address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],k[0-3][4],...}
//         r3     arg3  kernel size
//         sp     arg4  output address output                 : {i0k0  i1k0  i2k0  i3k0}
//                                     output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                     output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                     output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         sp+0x4 arg5  output xy 
//         sp+0x8 arg6  activation flag   relu layers is integrated after convolution
//
// output: no
//
// q0   4S kernel data { k3  k2  k1  k0 }[0]
// q1   4S kernel data { k3  k2  k1  k0 }[1]
// q2   4S kernel data { k3  k2  k1  k0 }[2]
// q3   4S kernel data { k3  k2  k1  k0 }[3]
// q4-q7  not used
// q8   4S input data  { i3  i2  i1  i0 }[0]
// q9   4S input data  { i3  i2  i1  i0 }[1]
// q10  4S input data  { i3  i2  i1  i0 }[2]
// q11  4S input data  { i3  i2  i1  i0 }[3]
// q12  dot product for {i3k0, i2k0, i1k0, i0k0}
// q13  dot product for {i3k1, i2k1, i1k1, i0k1}
// q14  dot product for {i3k2, i2k2, i1k2, i0k2}
// q15  dot product for {i3k3, i2k3, i1k3, i0k3}




	.section .text, "ax"
	.align 5

	.type sgemm_4x4_a17 STT_FUNC
	.global sgemm_4x4_a17
	.hidden sgemm_4x4_a17

sgemm_4x4_a17:
	teq		r0, #0x0		// biases address = nullptr?
	beq		non_biases

	// have biases
	vld2.32		{d24[],d26[]}, [r0]!
	vmov		d25, d24
	vmov		d27, d26
	vld2.32		{d28[],d30[]}, [r0]
	vmov		d29, d28
	vmov		d31, d30
	b		convolution_start

non_biases:
	vmov.i64	q12, #0x0
	vmov.i64	q13, #0x0
	vmov.i64	q14, #0x0
	vmov.i64	q15, #0x0

convolution_start:
	cmp		r3, #0x4
	blt		loop4_end
	lsr		r0, r3, #0x2		// kernel_size / 4

// main loop    each loop generate dot prodcut for 4x4x4SFP
loop4:
	vldr		d0,  [r2]
	vldr		d16, [r1]
	vldr		d17, [r1, #0x8]
	vldr		d1,  [r2, #0x8]
	subs		r0, r0, #0x1
	vmla.f32	q12, q8, d0[0]
	vldr		d18, [r1, #0x10]
	vmla.f32	q13, q8, d0[1]
	vldr		d19, [r1, #0x18]
	vmla.f32	q14, q8, d1[0]
	vldr		d2,  [r2, #0x10]
	vmla.f32	q15, q8, d1[1]
	vldr		d3,  [r2, #0x18]
	vmla.f32	q12, q9, d2[0]
	vldr		d20, [r1, #0x20]
	vmla.f32	q13, q9, d2[1]
	vldr		d21, [r1, #0x28]
	vmla.f32	q14, q9, d3[0]
	vldr		d4,  [r2, #0x20]
	vmla.f32	q15, q9, d3[1]
	vldr		d5,  [r2, #0x28]
	vmla.f32	q12, q10,d4[0]
	vldr		d22, [r1, #0x30]
	vmla.f32	q13, q10,d4[1]
	vldr		d23, [r1, #0x38]
	vmla.f32	q14, q10,d5[0]
	vldr		d6,  [r2, #0x30]
	vmla.f32	q15, q10,d5[1]
	vldr		d7,  [r2, #0x38]
	vmla.f32	q12, q11,d6[0]
	vmla.f32	q13, q11,d6[1]
	pld		[r2, #0x180]
	add		r2, r2, #0x40
	vmla.f32	q14, q11,d7[0]
	pld		[r1, #0x180]
	add		r1, r1, #0x40
	vmla.f32	q15, q11,d7[1]
	bne		loop4

loop4_end:
	ldr		r0, [sp, #0x8]
	ands		r3, r3, #0x3
	beq		activation

loop1:
	vldm		r1!, {d16 - d17}	// i[3-0]0
	vldm		r2!, {d0  -  d1}	// k[3-0]0
	subs		r3, r3, #0x1
	vmla.f32	q12, q8, d0[0]
	vmla.f32	q13, q8, d0[1]
	vmla.f32	q14, q8, d1[0]
	vmla.f32	q15, q8, d1[1]
	bne		loop1

activation:
	cmp		r0, #0x0
        vdup.32         q3, r0
	ldrd		r0, r1, [sp]		// r0 = output_address r1 = output_xy
	
        blt		save_result

	vmov.i64	q2, #0x0
	vmax.f32	q12, q12, q2
	vmax.f32	q13, q13, q2
	vmax.f32	q14, q14, q2
	vmax.f32	q15, q15, q2

    beq         save_result

	vcvt.f32.s32    q3, q3
	vmin.f32	q12, q12, q3
	vmin.f32	q13, q13, q3
	vmin.f32	q14, q14, q3
	vmin.f32	q15, q15, q3

save_result:
    ldr         r2, [sp, #0x0c]
    teq         r2, #0x0
    beq         save_result_nchw
    
	add		r2, r0, r1, LSL #3
	add		r3, r2, r1, LSL #2
	add		r1, r0, r1, LSL #2
    
    vst4.32    {d24[0], d26[0], d28[0], d30[0]}, [r0]!
    vst4.32    {d24[1], d26[1], d28[1], d30[1]}, [r1]!
    vst4.32    {d25[0], d27[0], d29[0], d31[0]}, [r2]!
    vst4.32    {d25[1], d27[1], d29[1], d31[1]}, [r3]!
    b           end
    
save_result_nchw:
	add		r2, r0, r1, LSL #3
	add		r3, r2, r1, LSL #2
	add		r1, r0, r1, LSL #2

	vstm		r0, {d24,d25}
	vstm		r1, {d26,d27}
	vstm		r2, {d28,d29}
	vstm		r3, {d30,d31}


end:
	bx	lr

	.end
