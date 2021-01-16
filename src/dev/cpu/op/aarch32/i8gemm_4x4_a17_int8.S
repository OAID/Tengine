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
// optimised for Cortex-A17 pipeline 17 cycle per loop (4*4*4 dot product)
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

        .type i8gemm_4x4_a17_int8 STT_FUNC
        .global i8gemm_4x4_a17_int8
        .hidden i8gemm_4x4_a17_int8
i8gemm_4x4_a17_int8:
        push            {r4 - r5}
	vpush		{d8 - d15}
	cmp		r3, #0x4
	vmov.i64	q4, #0x0
	vmov.i64	q5, #0x0
	vmov.i64	q6, #0x0
	vmov.i64	q7, #0x0

	vldr		d0, [r1]
	vldr		d2, [r2]
	blt		loop4_end
	lsr		r12,r3, #0x2	// kernel_size / 4

// main loop    each loop generate 4x4x4 dot product
loop4:
	vmull.s8	q2, d0, d2
	vldr		d1, [r1, #0x8]
	vrev32.16	d0, d0
	vldr		d3, [r2, #0x8]
	subs		r12, r12, #0x1
	vmlal.s8	q2, d1, d3
	add		r2, r2, #0x10
	vrev32.16	d1, d1
	vmull.s8	q3, d0, d2
	vrev64.32	d0, d0
	vpadal.s16	q4, q2
	add		r1, r1, #0x10
	vmlal.s8	q3, d1, d3
	vrev64.32	d1, d1
	vmull.s8	q2, d0, d2
	vrev32.16	d0, d0
	vpadal.s16	q5, q3
	pld		[r2, #0x80]
	vmlal.s8	q2, d1, d3
	vrev32.16	d1, d1
	vmull.s8	q3, d0, d2
	vpadal.s16	q7, q2
	pld		[r1, #0x60]
	vmlal.s8	q3, d1, d3
	vldr		d0, [r1]
	vldr		d2, [r2]
	vpadal.s16	q6, q3
	bne		loop4

loop4_end:
	ands		r3, r3, #0x3
	beq		add_bias

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

add_bias:
    // load and add biases
    teq		r0, #0x0
    beq		to_int8
    vldm	r0, {d0,d1}
    vadd.s32	q4, q4, q0	
    vadd.s32	q5, q5, q0	
    vadd.s32	q6, q6, q0	
    vadd.s32	q7, q7, q0	

to_int8:
    // convert result to sp and multiply with scale
    ldr     	r2, [sp, #0x4c]		// r2 = scale address  r3 = output_xy
    //vmov        s0, r2
    vldr        d0, [r2]
    vldr        d1, [r2, #0x8]
    vqrdmulh.s32    q4, q4, q0 //d0[0] 
    vqrdmulh.s32    q5, q5, q0 //d0[0] 
    vqrdmulh.s32    q6, q6, q0 //d0[0] 
    vqrdmulh.s32    q7, q7, q0 //d0[0] 

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

    vmax.s32	q4, q4, q8
    vmax.s32	q5, q5, q8
    vmax.s32	q6, q6, q8
    vmax.s32	q7, q7, q8

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
    //vst4.32	{d8[0], d10[0], d12[0], d14[0]}, [r0]
    vmov        r4, s16
    strb        r4, [r0]
    vmov        r4, s20
    strb        r4, [r0, #0x1]
    vmov        r4, s24
    strb        r4, [r0, #0x2]
    vmov        r4, s28
    strb        r4, [r0, #0x3]

    //vstr	s21, [r1]
    //vstr	s17, [r1, #0x4]
    //vstr	s29, [r1, #0x8]
    //vstr	s25, [r1, #0xc]
    vmov        r4, s21
    strb        r4, [r1]
    vmov        r4, s17
    strb        r4, [r1, #0x1]
    vmov        r4, s29
    strb        r4, [r1, #0x2]
    vmov        r4, s25
    strb        r4, [r1, #0x3]
    add		r0, r1, r3, LSL #1	
    
    //vstr	s26, [r2]
    //vstr	s30, [r2, #0x4]
    //vstr	s18, [r2, #0x8]
    //vstr	s22, [r2, #0xc]
    vmov        r4, s26
    strb        r4, [r2]
    vmov        r4, s30
    strb        r4, [r2, #0x1]
    vmov        r4, s18
    strb        r4, [r2, #0x2]
    vmov        r4, s22
    strb        r4, [r2, #0x3]
    
    //vstr	s31, [r0]
    //vstr	s27, [r0, #0x4]
    //vstr	s23, [r0, #0x8]
    //vstr	s19, [r0, #0xc]
    vmov        r4, s31
    strb        r4, [r0]
    vmov        r4, s27
    strb        r4, [r0, #0x1]
    vmov        r4, s23
    strb        r4, [r0, #0x2]
    vmov        r4, s19
    strb        r4, [r0, #0x3]
    
    b       end

indirect_save:
    vstm	r0, {d8-d15}

end:
    vpop	{d8 - d15}
    pop         {r4 - r5}
    bx	        lr

    .end
