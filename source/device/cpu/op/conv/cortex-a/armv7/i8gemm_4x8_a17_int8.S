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
// 4*8 INT8 matric multiplication
//
//    --              --      --               --     --               --         --                   --
//    | i0 - - - - - - |      |  k0  k1  .   k7 |     |  b0  b1  .   b7 |         | i0k0 i0k1 ..   i0k8 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   b7 |         | i1k0 i1k1 ..   i1k8 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                     |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   b7 |         | i2k0 i2k1 ..   i2k8 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   b7 |         | i3k0 i3k1 ..   i3k8 |
//    --              --      --               --     --               --         --                   --
//      input 4 x p             kernel p x 8             biases 4 x 8                  output 4 x 8          p = kernel size
//
//
// optimised for Cortex-A17 pipeline ?? cycle per loop (4*8*4 dot product)
// might load 16 byte input more to get better performance
//
// input:
//         r0     arg0  biases address {b0,b1,b2,b3,b4,b5,b6,b7} nullptr means no biases
//         r1     arg1  input  address {i[0-3][0-1],i[0-3][2-3],i[0-3][4-5],i[0-3][6-7],...}
//         r2     arg2  kernel address {k[0-7][0-1],k[0-7][2-3],k[0-7][4-5],k[0-7][6-7],...}
//         r3     arg3  kernel size need to be even number
//         sp     arg4  output address 
//                       indirect save:{i0k0,i1k1,i2k2,i3k3, i1k0,i0k1,i3k2,i2k3, i2k0,i3k1,i0k2,i1k3, i3k0,i2k1,i1k2,i0k3}
//                                     {i0k4,i1k5,i2k6,i3k7, i1k4,i0k5,i3k6,i2k7, i2k4,i3k5,i0k6,i1k7, i3k4,i2k5,i1k6,i0k7}
//                         direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                      output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                      ..
//                                      output + ouput_xy * 7 :  {i0k7  i1k7  i2k7  i3k7}
//         sp+0x4 arg5  scale address
//         sp+0x8 arg6  output xy
//         sp+0xc arg7  activation flag  relu layers is integrated after convolution
//         sp+0x10 arg8  layout
//
// output: no
//
// 1. {i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// 2. {i2[1-0], i3[1-0], i0[1-0], i1[1-0]} VREV32.16 V0
// 3. {i1[1-0], i0[1-0], i3[1-0], i2[1-0]} VREV64.32 V0
// 4. {i0[1-0], i1[1-0], i2[1-0], i3[1-0]} VREV64.16 V0
//
// q0  dot product {i3k3, i2k2, i1k1, i0k0}
// q1  dot product {i2k3, i3k2, i0k1, i1k0}
// q2  dot product {i1k3, i0k2, i3k1, i2k0}
// q3  dot product {i0k3, i1k2, i2k1, i3k0}
// q4  dot product {i3k7, i2k6, i1k5, i0k4}
// q5  dot product {i2k7, i3k6, i0k5, i1k4}
// q6  dot product {i1k7, i0k6, i3k5, i2k4}
// q7  dot product {i0k7, i1k6, i2k5, i3k4}
// d16 8byte input {i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// d17 8byte input {i3[3-2], i2[3-2], i1[3-2], i0[3-2]}
// d18 8byte input {i2[1-0], i3[1-0], i0[1-0], i1[1-0]}
// d19 8byte input {i2[3-2], i3[3-2], i0[3-2], i1[3-2]}
// d20 8byte input {i1[1-0], i0[1-0], i3[1-0], i2[1-0]}
// d21 8byte input {i1[3-2], i0[3-2], i3[3-2], i2[3-2]}
// d22 8byte input {i0[1-0], i1[1-0], i2[1-0], i3[1-0]}
// d23 8byte input {i0[3-2], i1[3-2], i2[3-2], i3[3-2]}
// d24 8byte kernel{k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// d25 8byte kernel{k7[1-0], k6[1-0], k5[1-0], k4[1-0]}
// d26 8byte kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2]}
// d27 8byte kernel{k7[3-2], k6[3-2], k5[3-2], k4[3-2]}
// q14 q15 temp reulsts

        .section .text,"ax"
        .align 5

        .type i8gemm_4x8_a17_int8 STT_FUNC
        .global i8gemm_4x8_a17_int8
        .hidden i8gemm_4x8_a17_int8
i8gemm_4x8_a17_int8:
        push            {r4 - r5}
	vpush		{d8 - d15}
	cmp		r3, #0x4
	vmov.i64	q0, #0x0
	vmov.i64	q1, #0x0
	vmov.i64	q2, #0x0
	vmov.i64	q3, #0x0
	vmov.i64	q4, #0x0
	vmov.i64	q5, #0x0
	vmov.i64	q6, #0x0
	vmov.i64	q7, #0x0

	lsr		r12,r3, #0x2	// kernel_size / 4
	vldm		r1, {d16-d17}
	blt		loop4_end

// main loop    each loop generate 4x8x4 dot product
loop4:
	vldr		d24, [r2]
	vldr		d26, [r2,#0x10]
	vrev32.16	q9, q8
	vldr		d25, [r2,#0x8]
	vrev64.32	q10,q8
	vldr		d27, [r2,#0x18]
	vrev64.16	q11, q8

	vmull.s8	q14, d16, d24
	vmlal.s8	q14, d17, d26
	subs		r12, r12, #0x1
	vmull.s8	q15, d16, d25
	vmlal.s8	q15, d17, d27
	vpadal.s16	q0,  q14

	vmull.s8	q14, d18, d24
	pld		[r1, #0x80]
	vmlal.s8	q14, d19, d26
	vpadal.s16	q4,  q15
	vmull.s8	q15, d18, d25
	vmlal.s8	q15, d19, d27
	vpadal.s16	q1,  q14

	vmull.s8	q14, d20, d24
	pld		[r2, #0x100]
	vmlal.s8	q14, d21, d26
	vpadal.s16	q5,  q15
	vmull.s8	q15, d20, d25
	add		r1, r1, #0x10
	vmlal.s8	q15, d21, d27
	vpadal.s16	q2,  q14

	vmull.s8	q14, d22, d24
	vmlal.s8	q14, d23, d26
	vpadal.s16	q6,  q15
	vmull.s8	q15, d22, d25
	add		r2, r2, #0x20
	vmlal.s8	q15, d23, d27
	vldm		r1, {d16-d17}
	vpadal.s16	q3,  q14
	vpadal.s16	q7,  q15

	bne		loop4

loop4_end:
	ands		r3, r3, #0x3
	beq		add_bias

// final 2 data
	vldm		r2, {d24-d25}
	vrev32.16	d18, d16
	vrev64.32	d20, d16
	vrev64.16	d22, d16

	vmull.s8	q14, d16, d24
	vmull.s8	q15, d16, d25
	vpadal.s16	q0,  q14
	vpadal.s16	q4,  q15

	vmull.s8	q14, d18, d24
	vmull.s8	q15, d18, d25
	vpadal.s16	q1,  q14
	vpadal.s16	q5,  q15

	vmull.s8	q14, d20, d24
	vmull.s8	q15, d20, d25
	vpadal.s16	q2,  q14
	vpadal.s16	q6,  q15

	vmull.s8	q14, d22, d24
	vmull.s8	q15, d22, d25
	vpadal.s16	q3,  q14
	vpadal.s16	q7,  q15

add_bias:
    // load and add biases
    teq		r0, #0x0
    beq		to_int8
    vldm	r0, {d16-d19}
    vadd.s32	q0, q0, q8	
    vadd.s32	q1, q1, q8	
    vadd.s32	q2, q2, q8	
    vadd.s32	q3, q3, q8	
    vadd.s32	q4, q4, q9	
    vadd.s32	q5, q5, q9	
    vadd.s32	q6, q6, q9	
    vadd.s32	q7, q7, q9	

to_int8:
    // convert result to sp and multiply with scale
    ldr     	r2, [sp, #0x4c]		// r2 = scale address  r3 = output_xy
    //vmov	q8, q0
    //vmov        s0, r2
    vldr        d16, [r2]
    vldr        d17, [r2, #0x8]
    vqrdmulh.s32    q0, q0, q8 //d0[0] 
    vqrdmulh.s32    q1, q1, q8 //d0[0] 
    vqrdmulh.s32    q2, q2, q8 //d0[0] 
    vqrdmulh.s32    q3, q3, q8 //d0[0]
    vldr        d16, [r2, #0x10]
    vldr        d17, [r2, #0x18]
    vqrdmulh.s32    q4, q4, q8 //d0[0] 
    vqrdmulh.s32    q5, q5, q8 //d0[0] 
    vqrdmulh.s32    q6, q6, q8 //d0[0] 
    vqrdmulh.s32    q7, q7, q8 //d0[0] 
    //vmov	q0, q8
    
    ldrd        r2, r3, [sp, #0x58]
    vdup.s32    q8, r2
    vdup.s32    q9, r3
    vmov.i64    d24, #0x0
    vmov.s32    d25, d24
    ldr         r2, [sp, #0x54] 
    //vdup.s32    q11, r2
    vldr        d22, [r2]
    vldr        d23, [r2, #0x8]
    vmax.s32    q10, q11, q12
    vmin.s32    q11, q11, q12
 
    vshl.s32 q0, q0, q10
    vshl.s32 q1, q1, q10
    vshl.s32 q2, q2, q10
    vshl.s32 q3, q3, q10
    vrshl.s32 q0, q0, q11
    vrshl.s32 q1, q1, q11
    vrshl.s32 q2, q2, q11
    vrshl.s32 q3, q3, q11

    vldr        d22, [r2, #0x10]
    vldr        d23, [r2, #0x18]
    vmax.s32    q10, q11, q12
    vmin.s32    q11, q11, q12

    vshl.s32 q4, q4, q10
    vshl.s32 q5, q5, q10
    vshl.s32 q6, q6, q10
    vshl.s32 q7, q7, q10
    vrshl.s32 q4, q4, q11
    vrshl.s32 q5, q5, q11
    vrshl.s32 q6, q6, q11
    vrshl.s32 q7, q7, q11

    ldr         r3, [sp, #0x50]

activation:

    vmax.s32	q0, q0, q8
    vmax.s32	q1, q1, q8
    vmax.s32	q2, q2, q8
    vmax.s32	q3, q3, q8
    vmax.s32	q4, q4, q8
    vmax.s32	q5, q5, q8
    vmax.s32	q6, q6, q8
    vmax.s32	q7, q7, q8

    vmin.s32	q0, q0, q9
    vmin.s32	q1, q1, q9
    vmin.s32	q2, q2, q9
    vmin.s32	q3, q3, q9
    vmin.s32	q4, q4, q9
    vmin.s32	q5, q5, q9
    vmin.s32	q6, q6, q9
    vmin.s32	q7, q7, q9

save_result:
    ldr		r0, [sp, #0x48]			// r0 = output address
    teq		r3, #0x0
    beq		indirect_save

    add		r1, r0, r3

    add		r2, r0, r3, LSL #1
    //output  r0, r1, r2, r0, r1, r2, r0, r1
    //vst4.32		{d0[0], d2[0], d4[0], d6[0]}, [r0]
    vmov        r4, s0
    strb        r4, [r0]
    vmov        r4, s4
    strb        r4, [r0, #0x1]
    vmov        r4, s8
    strb        r4, [r0, #0x2]
    vmov        r4, s12
    strb        r4, [r0, #0x3]
    
    //vstr		s5,  [r1]
    //vstr		s1,  [r1, #0x4]
    //vstr		s13, [r1, #0x8]
    //vstr		s9,  [r1, #0xc]
    vmov        r4, s5
    strb        r4, [r1]
    vmov        r4, s1
    strb        r4, [r1, #0x1]
    vmov        r4, s13
    strb        r4, [r1, #0x2]
    vmov        r4, s9
    strb        r4, [r1, #0x3]
    add		r0, r1, r3, LSL #1	
    
    //vstr		s10, [r2]
    //vstr		s14, [r2, #0x4]
    //vstr		s2,  [r2, #0x8]
    //vstr		s6,  [r2, #0xc]
    vmov        r4, s10
    strb        r4, [r2]
    vmov        r4, s14
    strb        r4, [r2, #0x1]
    vmov        r4, s2
    strb        r4, [r2, #0x2]
    vmov        r4, s6
    strb        r4, [r2, #0x3]
    add		r1, r2, r3, LSL #1	
    
    //vstr		s15, [r0]
    //vstr		s11, [r0, #0x4]
    //vstr		s7,  [r0, #0x8]
    //vstr		s3,  [r0, #0xc]
    vmov        r4, s15
    strb        r4, [r0]
    vmov        r4, s11
    strb        r4, [r0, #0x1]
    vmov        r4, s7
    strb        r4, [r0, #0x2]
    vmov        r4, s3
    strb        r4, [r0, #0x3]
    add		r2, r0, r3, LSL #1	
    
    //vst4.32		{d8[0],d10[0],d12[0],d14[0]}, [r1]
    vmov        r4, s16
    strb        r4, [r1]
    vmov        r4, s20
    strb        r4, [r1, #0x1]
    vmov        r4, s24
    strb        r4, [r1, #0x2]
    vmov        r4, s28
    strb        r4, [r1, #0x3]
    add		r0, r1, r3, LSL #1	
    
    //vstr		s21, [r2]
    //vstr		s17, [r2, #0x4]
    //vstr		s29, [r2, #0x8]
    //vstr		s25, [r2, #0xc]
    vmov        r4, s21
    strb        r4, [r2]
    vmov        r4, s17
    strb        r4, [r2, #0x1]
    vmov        r4, s29
    strb        r4, [r2, #0x2]
    vmov        r4, s25
    strb        r4, [r2, #0x3]
    add		r1, r2, r3, LSL #1	
    
    //vstr		s26, [r0]
    //vstr		s30, [r0, #0x4]
    //vstr		s18, [r0, #0x8]
    //vstr		s22, [r0, #0xc]
    vmov        r4, s26
    strb        r4, [r0]
    vmov        r4, s30
    strb        r4, [r0, #0x1]
    vmov        r4, s18
    strb        r4, [r0, #0x2]
    vmov        r4, s22
    strb        r4, [r0, #0x3]
    
    //vstr		s31, [r1]
    //vstr		s27, [r1, #0x4]
    //vstr		s23, [r1, #0x8]
    //vstr		s19, [r1, #0xc]
    vmov        r4, s31
    strb        r4, [r1]
    vmov        r4, s27
    strb        r4, [r1, #0x1]
    vmov        r4, s23
    strb        r4, [r1, #0x2]
    vmov        r4, s19
    strb        r4, [r1, #0x3]
    
    b		end

indirect_save:
    vstm	r0, {d0-d15}

end:
    vpop	{d8 - d15}
    pop     {r4 - r5}
    bx	lr

    .end
