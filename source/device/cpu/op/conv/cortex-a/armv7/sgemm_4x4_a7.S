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
// optimised for Cortex-A7 pipeline 89 cycle per loop (4*4*4 dot product)
//
// input:
//         r0     arg0  biases address {b0,b1,b2,b3}   nullptr means no biases
//         r1     arg1  input  address {i[0-3][0],i[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         r2     arg2  kernel address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],k[0-3][4],...}
//         r3     arg3  kernel size
//         sp     arg4  output address
//                      indirect save: output {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3],i[0-3]k[4]..}
//                        direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                     output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                     output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                     output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         sp+0x4 arg5  output xy
//         sp+0x8 arg6  activation flag   relu layers is integrated after convolution
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
// d14 2s kernel data  { k1 |  k0 }
// d15 2s kernel data  { k3 |  k2 }

	.section .text, "ax"
	.align 5

	.type sgemm_4x4_a7 STT_FUNC
	.global sgemm_4x4_a7
	.hidden sgemm_4x4_a7

sgemm_4x4_a7:
	// context save
	vpush		{d8 - d15}
	teq		r0, #0x0		// biases address = nullptr?
	beq		non_biases

	// have biases
	vld2.32		{d0[],d2[]}, [r0]!
	vmov		d1, d0
	vmov		d3, d2
	vld2.32		{d4[],d6[]}, [r0]
	vmov		d5, d4
	vmov		d7, d6
	b		convolution_start

non_biases:
	vmov.i64	q0, #0x0
	vmov.i64	q1, #0x0
	vmov.i64	q2, #0x0
	vmov.i64	q3, #0x0

convolution_start:
	cmp		r3, #0x4
	blt		loop4_end
	lsr		r0, r3, #0x2		// kernel_size / 4

// main loop    each loop generate dot prodcut for 4x4x4SFP
loop4:
	vldr		d12,[r2]		// k10, k00
	vldm		r1!, {d8-d11}		// i[3-0][1-0]
	vldr		d13,[r2, #0x8]		// k30, k20
	vldr		d14,[r2, #0x10]		// k11, k01
	vldr		d15,[r2, #0x18]		// k31, k21
	vmla.f32	d0, d8, d12[0]
	vmla.f32	d1, d9, d12[0]
	vmla.f32	d2, d8, d12[1]
	vmla.f32	d3, d9, d12[1]
	vmla.f32	d4, d8, d13[0]
	vmla.f32	d5, d9, d13[0]
	vmla.f32	d6, d8, d13[1]
	vmla.f32	d7, d9, d13[1]
	vmla.f32	d0, d10,d14[0]
	vmla.f32	d1, d11,d14[0]
	vmla.f32	d2, d10,d14[1]
	vmla.f32	d3, d11,d14[1]
	vmla.f32	d4, d10,d15[0]
	vmla.f32	d5, d11,d15[0]
	vmla.f32	d6, d10,d15[1]
	vmla.f32	d7, d11,d15[1]
	pld		[r1, #0x220]
	pld		[r2, #0x240]
	vldr		d12,[r2, #0x20]		// k12, k02
	vldm		r1!, {d8-d11}		// i[3-0][3-2]
	vldr		d13,[r2, #0x28]		// k32, k22
	vldr		d14,[r2, #0x30]		// k13, k03
	vldr		d15,[r2, #0x38]		// k33, k23
	add		r2, r2, #0x40
	vmla.f32	d0, d8, d12[0]
	vmla.f32	d1, d9, d12[0]
	vmla.f32	d2, d8, d12[1]
	vmla.f32	d3, d9, d12[1]
	vmla.f32	d4, d8, d13[0]
	vmla.f32	d5, d9, d13[0]
	vmla.f32	d6, d8, d13[1]
	vmla.f32	d7, d9, d13[1]
	vmla.f32	d0, d10,d14[0]
	vmla.f32	d1, d11,d14[0]
	vmla.f32	d2, d10,d14[1]
	vmla.f32	d3, d11,d14[1]
	vmla.f32	d4, d10,d15[0]
	vmla.f32	d5, d11,d15[0]
	vmla.f32	d6, d10,d15[1]
	vmla.f32	d7, d11,d15[1]
	subs		r0, r0, #0x1
	bne		loop4

loop4_end:
	ldr		r0, [sp, #0x48]
	ands		r3, r3, #0x3
	beq		activation

loop1:
	vldm		r1!, {d8  - d9}		// i[3-0]0
	vldm		r2!, {d12 - d13}	// k[3-0]0
	vmla.f32	d0, d8, d12[0]
	vmla.f32	d1, d9, d12[0]
	vmla.f32	d2, d8, d12[1]
	vmla.f32	d3, d9, d12[1]
	vmla.f32	d4, d8, d13[0]
	vmla.f32	d5, d9, d13[0]
	vmla.f32	d6, d8, d13[1]
	vmla.f32	d7, d9, d13[1]
	subs		r3, r3, #0x1
	bne		loop1

activation:
	cmp		r0, #0x0
        vdup.32         d9, r0
	ldrd		r0, r1, [sp, #0x40]	// r0 = output_address r1 = output_xy
	
        blt		save_result

	vmov.i64	d8, #0x0
	vmax.f32	d0, d0, d8
	vmax.f32	d1, d1, d8
	vmax.f32	d2, d2, d8
	vmax.f32	d3, d3, d8
	vmax.f32	d4, d4, d8
	vmax.f32	d5, d5, d8
	vmax.f32	d6, d6, d8
	vmax.f32	d7, d7, d8

        beq		save_result

	vcvt.f32.s32    d9, d9
	vmin.f32	d0, d0, d9
	vmin.f32	d1, d1, d9
	vmin.f32	d2, d2, d9
	vmin.f32	d3, d3, d9
	vmin.f32	d4, d4, d9
	vmin.f32	d5, d5, d9
	vmin.f32	d6, d6, d9
	vmin.f32	d7, d7, d9

save_result:
    ldr         r2, [sp, #0x4c]
    teq         r2, #0x0
    beq         save_result_nchw
    
	add		r2, r0, r1, LSL #3
	add		r3, r2, r1, LSL #2
	add		r1, r0, r1, LSL #2
    
    vst4.32    {d0[0], d2[0], d4[0], d6[0]}, [r0]!
    vst4.32    {d0[1], d2[1], d4[1], d6[1]}, [r1]!
    vst4.32    {d1[0], d3[0], d5[0], d7[0]}, [r2]!
    vst4.32    {d1[1], d3[1], d5[1], d7[1]}, [r3]!
    b           end
    
save_result_nchw:

	add		r2, r0, r1, LSL #3
	add		r3, r2, r1, LSL #2
	add		r1, r0, r1, LSL #2
	vstm		r0, {d0,d1}
	vstm		r1, {d2,d3}
	vstm		r2, {d4,d5}
	vstm		r3, {d6,d7}

end:
	vpop		{d8 - d15}
	bx	lr

	.end
