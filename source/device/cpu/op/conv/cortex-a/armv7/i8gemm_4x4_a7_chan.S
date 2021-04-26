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
// 4*4 INT8 matric multiplication
//
//    --              --      --               --     --               --         --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  k3 |     |  b0  b1  b2  b3 |         | i0k0 i0k1 .. i0k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i1k0 i1k1 .. i1k3 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i2k0 i2k1 .. i2k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i3k0 i3k1 .. i3k3 |
//    --              --      --               --     --               --         --                 --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 4           p = kernel size
//
//
// optimised for Cortex-A7 pipeline 45 cycle per loop (4*4*4 dot product)
//
// input:
//         r0     arg0  biases address {b0, b1, b2, b3}  nullptr means no biases
//         r1     arg1  input  address {i[0-3][0-1],i[0-3][2-3],i[0-3][4-5],i[0-3][6-7],...}
//         r2     arg2  kernel address {k[0-3][0-1],k[0-3][2-3],k[0-3][4-5],k[0-3][6-7],...}
//         r3     arg3  kernel size need to be even number
//         sp     arg4  output address 
//                       indirect save:{i0k0,i1k1,i2k2,i3k3, i1k0,i0k1,i3k2,i2k3, i2k0,i3k1,i0k2,i1k3, i3k0,i2k1,i1k2,i0k3}
//                         direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                      output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                      output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                      output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         sp+0x4 arg5  scale address
//         sp+0x8 arg6  output xy
//         sp+0xc arg7  activation flag  relu layers is integrated after convolution
//         sp+0x10 arg8  layout
//
// output: no
//
// 1. {i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// 2. {i2[1-0], i3[1-0], i0[1-0], i1[1-0]} VREV32.16 V0 
// 3. {i0[1-0], i1[1-0], i2[1-0], i3[1-0]} VREV64.32 V1
// 4. {i1[1-0], i0[1-0], i3[1-0], i2[1-0]} VREV32.16 V2
//
// d0  8byte input {i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// d1  8byte input {i3[3-2], i2[3-2], i1[3-2], i0[3-2]}
// d2  8byte kernel{k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// d3  8byte kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2]} 
// d4-7 temp reulst 
// q4 dot product {i3k3, i2k2, i1k1, i0k0}
// q5 dot product {i2k3, i3k2, i0k1, i1k0}
// q6 dot product {i1k3, i0k2, i3k1, i2k0}
// q7 dot product {i0k3, i1k2, i2k1, i3k0}

        .section .text,"ax"
        .align 5

        .type i8gemm_4x4_a7_chan STT_FUNC
        .global i8gemm_4x4_a7_chan
        .hidden i8gemm_4x4_a7_chan
i8gemm_4x4_a7_chan:
	vpush		{d8 - d15}
	vmov.i64	q4, #0x0
	vmov.i64	q5, #0x0
	vmov.i64	q6, #0x0
	vmov.i64	q7, #0x0

	vldr		d0, [r1]
	vldr		d2, [r2]
	cmp		r3, #0x4
	blt		loop4_end
	lsr		r12,r3, #0x2	// kernel_size / 4

// main loop    each loop generate 4x4x4 dot product
loop4:
	vmull.s8	q2, d0, d2
	vldr		d1, [r1, #0x8]
	vldr		d3, [r2, #0x8]
	vrev32.16	d0, d0
	vmlal.s8	q2, d1, d3
	pld		[r2, #0x80]
	add		r2, r2, #0x10
	vrev32.16	d1, d1
	vmull.s8	q3, d0, d2
	vrev64.32	d0, d0
	vpadal.s16	q4, q2
	pld		[r1, #0x60]
	add		r1, r1, #0x10
	vmlal.s8	q3, d1, d3
	vrev64.32	d1, d1
	vmull.s8	q2, d0, d2
	vrev32.16	d0, d0
	vpadal.s16	q5, q3
	vmlal.s8	q2, d1, d3
	vrev32.16	d1, d1
	vmull.s8	q3, d0, d2
	vpadal.s16	q7, q2
	vmlal.s8	q3, d1, d3
	vldr		d0, [r1]
	vldr		d2, [r2]
	vpadal.s16	q6, q3
	subs		r12, r12, #0x1
	bne		loop4

loop4_end:
	ands		r3, r3, #0x3
	beq		int2fp

// final 2 data
	vrev32.16	d1, d0
	vmull.s8	q2, d0, d2
	vmull.s8	q3, d1, d2
	vrev64.32	d0, d1
	vpadal.s16	q4, q2
	vmull.s8	q2, d0, d2
	vrev32.16	d1, d0
	vpadal.s16	q5, q3
	vmull.s8	q3, d1, d2
	vpadal.s16	q7, q2
	vpadal.s16	q6, q3

int2fp:
	// convert result to sp and multiply with scale
    ldrd     	r2, r3, [sp, #0x44]		// r2 = scale address  r3 = output_xy
    vcvt.f32.s32	q4, q4
    vcvt.f32.s32	q5, q5
    vldm		r2, {d0, d1}		// scale
    vcvt.f32.s32	q6, q6
    vcvt.f32.s32	q7, q7
    vmul.f32	q4, q4, q0
    vmul.f32	q5, q5, q0
    vmul.f32	q6, q6, q0
    vmul.f32	q7, q7, q0

	// load and add biases
	teq		r0, #0x0
	beq		activation
    vldm		r0, {d0,d1}
    vadd.f32	q4, q4, q0	
    vadd.f32	q5, q5, q0	
    vadd.f32	q6, q6, q0	
    vadd.f32	q7, q7, q0	

activation:
	ldr		r0, [sp, #0x4c]
	cmp		r0, #0x0

    blt		save_result

    vmov.i64	q0, #0x0
	vmax.f32	q4, q4, q0
	vmax.f32	q5, q5, q0
	vmax.f32	q6, q6, q0
	vmax.f32	q7, q7, q0
    
    beq             save_result
    
    vdup.32         q0, r0
    vcvt.f32.s32    q0, q0 
    vmin.f32	q4, q4, q0
	vmin.f32	q5, q5, q0
	vmin.f32	q6, q6, q0
	vmin.f32	q7, q7, q0

save_result:
	ldr		r0, [sp, #0x40]			// r0 = output address 
	ldr		r2, [sp, #0x50]			// r2 = layout
	teq		r3, #0x0
    beq		indirect_save
	lsl		r3, r3, #2
	add		r1, r0, r3
    teq     r2, #0x0
    beq     save_result_nchw

	add		r2, r0, r3, LSL #1

	vstr		s16, [r0]
    add     r3, r2, r3
	vstr		s20, [r1]
	vstr		s24, [r2]
	vstr		s28, [r3]
	vstr		s21, [r0, #0x4]
	vstr		s17, [r1, #0x4]
	vstr		s29, [r2, #0x4]
	vstr		s25, [r3, #0x4]

	vstr		s26, [r0, #0x8]
	vstr		s30, [r1, #0x8]
	vstr		s18, [r2, #0x8]
	vstr		s22, [r3, #0x8]

	vstr		s31, [r0, #0xc]
	vstr		s27, [r1, #0xc]
	vstr		s23, [r2, #0xc]
	vstr		s19, [r3, #0xc]

    b   end

save_result_nchw:
	add		r2, r0, r3, LSL #1

	vst4.32		{d8[0], d10[0], d12[0], d14[0]}, [r0]
	add		r0, r1, r3, LSL #1	

	vstr		s21, [r1]
	vstr		s17, [r1, #0x4]
	vstr		s29, [r1, #0x8]
	vstr		s25, [r1, #0xc]

	vstr		s26, [r2]
	vstr		s30, [r2, #0x4]
	vstr		s18, [r2, #0x8]
	vstr		s22, [r2, #0xc]
	
	vstr		s31, [r0]
	vstr		s27, [r0, #0x4]
	vstr		s23, [r0, #0x8]
	vstr		s19, [r0, #0xc]

    b   end

indirect_save:
	vstm		r0, {d8-d15}

end:
	vpop		{d8 - d15}
	bx		lr

        .end
