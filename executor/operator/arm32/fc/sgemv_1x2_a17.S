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
//                            --        --
//                            |  k0  k1  |
//                            |  .   .   |
//    --              --      |  .   .   |     --        --         --        --
//    | i0 - - - - - - |  x   |  .   .   |  +  |  b0  b1  |     =   | i0k0 i0k1 |
//    --              --      |  .   .   |     --        --         --         --
//                            |  .   .   |
//                            |  .   .   |
//                            --        --
//      input 1 x p           kernel p x 2        biases 1 x 2       output 1 x 2           p = kernel size
//
//
// optimised for Cortex-A17 pipeline 7 cycle per loop (1*2*4 dot product)
// the bottleneck is memory bandwidth
//
// input:
//         r0     arg0   biases start address      {b0, b1}   nullptr means no biases
//         r1     arg1   input data start address  {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, ...}
//         r2     arg2   kernel data start address {k00, k10, k01, k11, k02, k03, ...}
//         r3     arg3   kernel size
//         sp     arg4   output data save address  {ik0, ik1}
//
// output: no
//
// register definition
//
// d0  dot product for {ik1, ik0}
// d1  dot product for {ik1, ik0}
// d2  2s input  data  {i1 | i0 }
// d3  2s input  data  {i3 | i2 }
// d4  2S kernel data  {k1 | k0 }
// d5  2S kernel data  {k1 | k0 }
// d6  2S kernel data  {k1 | k0 }
// d7  2S kernel data  {k1 | k0 }
// d8  dot product for {ik1, ik0}
// d9  dot product for {ik1, ik0}
// d10~d15 not used

	.section .text, "ax"
	.align 5

	.type sgemv_1x2_a17 STT_FUNC
	.global sgemv_1x2_a17
	.hidden sgemv_1x2_a17

sgemv_1x2_a17:
	teq		r0, #0x0		// have_biases flag
	vmov.i64	d0, #0x0
	vpush		{d8, d9}
	vldrne		d0, [r0]		

	cmp		r3, #0x4
	blt		loop4_end
	vmov.i64	d1, #0x0
	vmov.i64	d8, #0x0
	vmov.i64	d9, #0x0
	lsr		r0, r3, #0x2		// kernel_size / 2

// main loop    each loop generate dot prodcut for 1x2x4SP
loop4:
        vldm            r1, {d2 - d3}           // i[3-0]
        vldm            r2, {d4 - d7}           // k[1-0][3-0]
        subs            r0, r0, #0x1
        vmla.f32        d0, d4, d2[0]
        vmla.f32        d1, d5, d2[1]
        pld             [r1, #0x40]
        add             r1, r1, #0x10
        pld             [r2, #0x80]
        add             r2, r2, #0x20
        vmla.f32        d8, d6, d3[0]
        vmla.f32        d9, d7, d3[1]
        bne             loop4

loop4_end:
	vadd.f32	d0, d0, d1
	vadd.f32	d8, d8, d9
	ands		r3, #0x3
	ldr		r0, [sp, #0x10]		// output save address
	vadd.f32	d0, d0, d8
	beq		save_result

loop1:
	vldm		r2!, {d4}		// k10, k00
	vldm		r1!, {s2} 		// i0
	subs		r3, r3, #0x1
	vmla.f32	d0, d4, d1[0]
	bne		loop1

save_result:
	vstr 		d0, [r0]
	vpop		{d8, d9}

	bx		lr

	.end
