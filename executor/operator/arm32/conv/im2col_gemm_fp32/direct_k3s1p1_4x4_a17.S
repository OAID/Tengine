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
 * Author: haoluo@openailab.com
 */
 
// input:
//         r0     arg0  biases address {b0,b1,b2,b3}   nullptr means no biases
//         r1     arg1  input  address 
//         r2     arg2  kernel address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],k[0-15][4],...}
//         r3     arg4  output address
//                        direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                     output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                     output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                     output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         sp     arg3  input_chan
//         sp+0x4 	  arg5  output x
//		   sp+0x8 	  arg6	output y
//         sp+0xc	  arg7  activation flag   relu layers is integrated after convolution
//
// output: no
//
//	r4	loop_chan
//	r5  loop_w
//	r6  tmp
//  r7  input_cur
//  r8  kernel_cur
//  r9  output_cur
//  r10 h*w*4
//  r11 loop_w_less
//	r12 loop_h
//
//
// v0   4S data of input0   {i3   i2   i1   i0} 
// v1   4S data of input1   {i7   i6   i5   i4}
// v2   4S data of input1   {ic   ia   i9   i8}
//
// q8   4S kernal data      {k3 | k2 | k1 | k0}
// q9   4S kernal data      {k7 | k6 | k5 | k4}
//
// q10  dot product for {i0k3  i0k2  i0k1  i0k0}
// q11  dot product for {i1k3  i1k2  i1k1  i1k0}
// q12  dot product for {i2k3  i2k2  i2k1  i2k0}
// q13  dot product for {i3k3  i3k2  i3k1  i3k0}
//
	.section .text, "ax"
	.align 5

	.type direct_k3s1p1_4x4_a17 STT_FUNC
	.global direct_k3s1p1_4x4_a17
	.hidden direct_k3s1p1_4x4_a17

direct_k3s1p1_4x4_a17:
	push		{r4,r5,r6,r7,r8,r9,r10,r11,r12,lr}
	pld		[r2, #48]
	pld		[r1]
	ldr  r4, [sp, #40]
	ldr  r6, [sp, #44] 
	ldr  r5, [sp, #48]    
	sub  r11, r6, #5      //input_w - 5
	sub  r12, r5, #2      //input_h - 2
	lsl  r6,  r6, #2
	mul  r10, r5, r6
	
	
//---------------------------------------------------   first  row ---------------------------------------
row_first_1:
	// biases_initial
	teq	    r0, #0x0
	beq     none_biases1
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	
	b	convolution_start1
	
none_biases1:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start1:
	mov  r7, r1
	mov  r8, r2

loop_channel_1:
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	pld		[r2, #336]
	pld		[r7, r10, lsl #1]
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	subs    r4,  r4, #1
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	add    r7,  r7,  r6
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #96]   // k6
	vldr   d7, [r8, #104]
	sub    r7,  r7,  r6
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #112]   // k7
	vldr   d7, [r8, #120]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #128]   // k8
	vldr   d7, [r8, #136]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	bne  loop_channel_1
	
activation_1:
	ldr	  r6, [sp, #52]
	mov   r9,  r3
	cmp   r6, #0
	blt   save_result_1

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_1

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_1:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	
	add  r9, r3, #16
	add  r7, r1, #12
	mov  r8, r2 		// kernel
	ldr  r4, [sp, #40]
	
row_loop_2:
	lsr  r5,  r11, #2
	teq  r5, #0
	beq  row_loop_less_2
	
row_loop4_start_2:
	teq	    r0, #0x0
	beq     none_biases2
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	b	convolution_start2
	
none_biases2:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start2:
	ldr  r6, [sp, #44]
	
loop_channel_2:
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	subs   r4,  r4, #1
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	add    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #96]   // k6
	vldr   d7, [r8, #104]
	sub    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #112]   // k7
	vldr   d7, [r8, #120]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #128]   // k8
	vldr   d7, [r8, #136]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	bne  loop_channel_2
	
activation_2:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_2

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_2

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_2:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #0x10
	add  r7,  r7,  #0x10
	
row_loop4_end_2:
	subs r5,  r5,  #1
	bne  row_loop4_start_2
	
	
row_loop_less_2:
	ands r5, r11, #3
	beq  row_end_3

row_loop1_start_2:
	teq	    r0, #0x0
	beq     none_biases2_1
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
	b	convolution_start2_1
	
none_biases2_1:
	vmov.i64	q8 , #0x0
	
convolution_start2_1:
	ldr  r6,  [sp, #44]
	
loop_channel_2_1:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #48]   // k3
	vldr   d7,  [r8, #56]
	vldr   d18, [r8, #64]   // k4
	vldr   d19, [r8, #72]
	vldr   d20, [r8, #80]   // k5
	vldr   d21, [r8, #88]
	add  r7,  r7,  r6, lsl  #2
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #96]   // k6
	vldr   d7,  [r8, #104]
	vldr   d18, [r8, #112]   // k7
	vldr   d19, [r8, #120]
	vldr   d20, [r8, #128]   // k8
	vldr   d21, [r8, #136]
	sub  r7,  r7,  r6, lsl  #2
	subs r4,  r4,  #1
	add  r8,  r8,  #144
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	add  r7,  r7,  r10
	
	bne  loop_channel_2_1
	
activation_2_1:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_2_1

	vmov.i64	q0, #0x0
	vmax.f32	q8, q8, q0

        beq   save_result_2_1

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8, q8, q0

save_result_2_1:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #4
	add  r7,  r7,  #4
	
row_loop1_end_2:
	subs r5,  r5,  #1
	bne  row_loop1_start_2
	
row_end_3:
	teq	    r0, #0x0
	beq     none_biases3
		vldr  d16,  [r0]
		vldr  d17,  [r0, #8]
	b	convolution_start3
	
none_biases3:
	vmov.i64	q8 , #0x0
	
convolution_start3:
	ldr  r6,  [sp,  #44]
	lsl  r6,  r6 ,   #2
		
loop_channel_3:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0 , [r7]
	add  r7,  r7,  r6
	vldr   d6 , [r8, #48]   // k3
	vldr   d7 , [r8, #56]
	vldr   d1 , [r7]
	vldr   d18, [r8, #64]   // k4
	vldr   d19, [r8, #72]
	vldr   d20, [r8, #96]   // k6
	vldr   d21, [r8, #104]
	vldr   d22, [r8, #112]   // k7
	vldr   d23, [r8, #120]
	sub  r7,  r7,  r6
	add  r8,  r8, #144
	subs r4,  r4,  #1
	add  r7,  r7,  r10
	vmla.f32	q8,  q3 , d0[0]
	vmla.f32	q8,  q9 , d0[1]
	vmla.f32	q8,  q10, d1[0]
	vmla.f32	q8,  q11, d1[1]
	
	bne  loop_channel_3
	
activation_3:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_3

	vmov.i64	q0, #0x0
	vmax.f32	q8, q8, q0

        beq   save_result_3

	vdup.32         q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8, q8, q0

save_result_3:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10
	sub  r9,  r9,  r10, lsl #2
	mov  r8,  r2			// kernel
	add  r9,  r9,  #4

//------------------------------------------middle  loop -------------------------------
	
	mov  r7,  r1
	mov  r8,  r2
	
middle_loop_start:
row_first_4:
	// biases_initial
	teq	    r0, #0
	beq     none_biases4
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	
	b	convolution_start4
	
none_biases4:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start4:
	ldr  r6, [sp, #44]
	ldr  r4, [sp, #40]

loop_channel_4:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #0]    // k0
	vldr   d7, [r8, #8]
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #16]   // k1
	vldr   d7, [r8, #24]
	subs   r4,  r4, #1
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #32]   // k2
	vldr   d7, [r8, #40]
	add    r7,  r7,  r6,  lsl #2
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	add    r7,  r7,  r6,  lsl #2
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #96]   // k6
	vldr   d7, [r8, #104]
	sub    r7,  r7,  r6, lsl #3
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #112]   // k7
	vldr   d7, [r8, #120]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #128]   // k8
	vldr   d7, [r8, #136]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	
	bne  loop_channel_4
	
activation_4:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_4

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_4

	vdup.32         q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_4:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #16
	add  r7,  r7,  #12

row_loop_5:
	lsr  r5,  r11, #2
	teq  r5, #0
	beq  row_loop_less_5
	
row_loop4_start_5:
	teq	    r0, #0x0
	beq     none_biases5
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	b	convolution_start5
	
none_biases5:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start5:
	ldr  r6, [sp, #44]
	
loop_channel_5:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8]   	   // k0
	vldr   d7, [r8, #8]
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #16]   // k1
	vldr   d7, [r8, #24]
	subs   r4,  r4, #1
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #32]   // k2
	vldr   d7, [r8, #40]
	add    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	add    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #96]   // k6
	vldr   d7, [r8, #104]
	sub    r7,  r7,  r6, lsl #3
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #112]   // k7
	vldr   d7, [r8, #120]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #128]   // k8
	vldr   d7, [r8, #136]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	bne  loop_channel_5
	
activation_5:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_5

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_5

	vdup.32         q0, r6
        vcvt.f32.s32    q0, q0
        vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_5:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #0x10
	add  r7,  r7,  #0x10
	
row_loop4_end_5:
	subs r5,  r5,  #1
	bne  row_loop4_start_5
	
	
row_loop_less_5:
	ands r5, r11, #3
	beq  row_end_6
	
row_loop1_start_5:
	teq	    r0, #0x0
	beq     none_biases5_1
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
	b	convolution_start5_1
	
none_biases5_1:
	vmov.i64	q8 , #0x0
	
convolution_start5_1:
	ldr  r6,  [sp, #44]
	
loop_channel_5_1:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #0]   // k0
	vldr   d7,  [r8, #8]
	vldr   d18, [r8, #16]   // k1
	vldr   d19, [r8, #24]
	vldr   d20, [r8, #32]   // k2
	vldr   d21, [r8, #40]
	add  r7,  r7,  r6, lsl  #2
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #48]   // k3
	vldr   d7,  [r8, #56]
	vldr   d18, [r8, #64]   // k4
	vldr   d19, [r8, #72]
	vldr   d20, [r8, #80]   // k5
	vldr   d21, [r8, #88]
	add  r7,  r7,  r6, lsl  #2
	subs r4,  r4,  #1
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	
	
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #96]   // k6
	vldr   d7,  [r8, #104]
	vldr   d18, [r8, #112]   // k7
	vldr   d19, [r8, #120]
	vldr   d20, [r8, #128]   // k8
	vldr   d21, [r8, #136]
	sub  r7,  r7,  r6, lsl  #3
	add  r8,  r8,  #144
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	add  r7,  r7,  r10

	bne  loop_channel_5_1
	
activation_5_1:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_5_1

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0

        beq   save_result_5_1

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0

save_result_5_1:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #4
	add  r7,  r7,  #4
	
row_loop1_end_5:
	subs r5,  r5,  #1
	bne  row_loop1_start_5
	

row_end_6:
	teq	    r0, #0x0
	beq     none_biases6
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
	b	convolution_start6
	
none_biases6:
	vmov.i64	q8 , #0x0
	
convolution_start6:
	ldr  r6,  [sp,  #44]
		
loop_channel_6:
	pld		[r8, #336]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d4, [r8]        // k0
	vldr   d5, [r8, #8]
	vldr   d6, [r8, #16]   // k1
	vldr   d7, [r8, #24]
	add  r7,  r7,  r6, lsl  #2
	vmla.f32	q8,  q2 , d0[0]
	vmla.f32	q8,  q3 , d0[1]
	
	vldr   d0, [r7]
	vldr   d4 , [r8, #48]   // k3
	vldr   d5 , [r8, #56]
	vldr   d6 , [r8, #64]   // k4
	vldr   d7 , [r8, #72]
	add  r7,  r7,  r6, lsl  #2
	subs r4,  r4,  #1
	vmla.f32	q8,  q2 , d0[0]
	vmla.f32	q8,  q3 , d0[1]
	
	vldr   d0, [r7]
	vldr   d4 , [r8, #96]   // k6
	vldr   d5 , [r8, #104]
	vldr   d6 , [r8, #112]   // k7
	vldr   d7 , [r8, #120]
	sub  r7,  r7,  r6, lsl  #3
	add  r8,  r8, #144
	vmla.f32	q8,  q2 , d0[0]
	vmla.f32	q8,  q3 , d0[1]
	add  r7,  r7,  r10
	
	bne  loop_channel_6
	
activation_6:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_6

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0

        beq   save_result_6

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0

save_result_6:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #4
	add  r7,  r7,  #8
	
middle_loop_end:
	subs r12,  r12,  #1
	bne  middle_loop_start

//------------------------------------------middle  end -------------------------------

row_first_7:
	// biases_initial
	teq	    r0, #0x0
	beq     none_biases7
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	
	b	convolution_start7
	
none_biases7:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start7:
	ldr  r6,  [sp,#44]

loop_channel_7:
	pld		[r8, #288]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8]   	   // k0
	vldr   d7, [r8, #8]
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #16]   // k1
	vldr   d7, [r8, #24]
	subs   r4,  r4, #1
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #32]   // k2
	vldr   d7, [r8, #40]
	add    r7,  r7,  r6,  lsl #2
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	sub    r7,  r7,  r6,  lsl #2
	vmla.f32	q9,  q3, d0[0]
	vmla.f32	q10, q3, d0[1]
	vmla.f32	q11, q3, d1[0]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	bne  loop_channel_7
	
activation_7:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_7

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_7

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_7:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	ldr  r4, [sp, #40]
	mov  r8,  r2 		// kernel
	mul  r6,  r4, r10
	sub  r9,  r9, r10, lsl #2
	sub  r7,  r7, r6
	add  r9,  r9, #0x10
	add  r7,  r7, #12 

row_loop_8:
	lsr  r5,  r11, #2
	teq  r5, #0
	beq  row_loop_less_8
	
row_loop4_start_8:
	teq	    r0, #0x0
	beq     none_biases8
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
		vmov.i64   q9,  q8
		vmov.i64   q10, q8
		vmov.i64   q11, q8
	b	convolution_start8
	
none_biases8:
	vmov.i64	q8 , #0x0
	vmov.i64	q9 , #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	
convolution_start8:
	ldr  r6, [sp, #44]
	
loop_channel_8:
	pld		[r8, #288]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8]   	   // k0
	vldr   d7, [r8, #8]
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #16]   // k1
	vldr   d7, [r8, #24]
	subs   r4,  r4, #1
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #32]   // k2
	vldr   d7, [r8, #40]
	add    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	
	vldr   d0, [r7]
	vldr   d1, [r7, #8]
	vldr   d2, [r7, #16]
	vldr   d6, [r8, #48]   // k3
	vldr   d7, [r8, #56]
	sub    r7,  r7,  r6, lsl #2
	vmla.f32	q8,  q3, d0[0]
	vmla.f32	q9,  q3, d0[1]
	vmla.f32	q10, q3, d1[0]
	vmla.f32	q11, q3, d1[1]
	vldr   d6, [r8, #64]   // k4
	vldr   d7, [r8, #72]
	add    r7,  r7,  r10
	vmla.f32	q8,  q3, d0[1]
	vmla.f32	q9,  q3, d1[0]
	vmla.f32	q10, q3, d1[1]
	vmla.f32	q11, q3, d2[0]
	vldr   d6, [r8, #80]   // k5
	vldr   d7, [r8, #88]
	add    r8,  r8,  #144
	vmla.f32	q8,  q3, d1[0]
	vmla.f32	q9,  q3, d1[1]
	vmla.f32	q10, q3, d2[0]
	vmla.f32	q11, q3, d2[1]
	bne  loop_channel_8
	
activation_8:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_8

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0
	vmax.f32	q9 , q9 , q0
	vmax.f32	q10, q10, q0
	vmax.f32	q11, q11, q0

        beq   save_result_8

	vdup.32         q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0
	vmin.f32	q9 , q9 , q0
	vmin.f32	q10, q10, q0
	vmin.f32	q11, q11, q0

save_result_8:

	vst4.32 {d16[0],d18[0],d20[0],d22[0]}, [r9], r10
	vst4.32 {d16[1],d18[1],d20[1],d22[1]}, [r9], r10
	vst4.32 {d17[0],d19[0],d21[0],d23[0]}, [r9], r10
	vst4.32 {d17[1],d19[1],d21[1],d23[1]}, [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #0x10
	add  r7,  r7,  #0x10
	
row_loop4_end_8:
	sub  r5,  r5,  #1
	teq  r5,  #0
	bne  row_loop4_start_8
	
	
row_loop_less_8:
	and  r5, r11, #3
	teq  r5, #0
	beq  row_end_9
	
row_loop1_start_8:
	teq	    r0, #0x0
	beq     none_biases8_1
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
	b	convolution_start8_1
	
none_biases8_1:
	vmov.i64	q8 , #0x0
	
convolution_start8_1:
	ldr  r6,  [sp, #44]
	
loop_channel_8_1:
	pld		[r8, #288]
	pld		[r7, r10, lsl #1]
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #0]   // k0
	vldr   d7,  [r8, #8]
	vldr   d18, [r8, #16]   // k1
	vldr   d19, [r8, #24]
	vldr   d20, [r8, #32]   // k2
	vldr   d21, [r8, #40]
	add  r7,  r7,  r6, lsl  #2
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	
	vldr   d0,  [r7]
	vldr   d1,  [r7, #8]
	vldr   d6,  [r8, #48]   // k3
	vldr   d7,  [r8, #56]
	vldr   d18, [r8, #64]   // k4
	vldr   d19, [r8, #72]
	vldr   d20, [r8, #80]   // k5
	vldr   d21, [r8, #88]
	sub  r7,  r7,  r6, lsl  #2
	subs r4,  r4,  #1
	add  r8,  r8,  #144
	vmla.f32	q8,  q3,  d0[0]
	vmla.f32	q8,  q9,  d0[1]
	vmla.f32	q8,  q10, d1[0]
	add  r7,  r7,  r10
	
	bne  loop_channel_8_1
	
activation_8_1:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_8_1

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0

        beq   save_result_8_1

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0

save_result_8_1:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10
	ldr  r4, [sp, #40]
	sub  r9,  r9,  r10, lsl #2
	mul  r6,  r4,  r10
	mov  r8,  r2			// kernel
	sub  r7,  r7,  r6
	add  r9,  r9,  #4
	add  r7,  r7,  #4
	
row_loop1_end_8:
	subs r5,  r5,  #1
	bne  row_loop1_start_8
	
row_end_9:
	teq	    r0, #0x0
	beq     none_biases9
		vldr  d16,  [r0,#0]
		vldr  d17,  [r0,#8]
	b	convolution_start9
	
none_biases9:
	vmov.i64	q8 , #0x0
	
convolution_start9:
	ldr  r6,  [sp,  #44]
		
loop_channel_9:
	pld		[r8, #288]
	pld		[r7, r10, lsl #1]
	vldr   d0, [r7]
	vldr   d6, [r8]   // k0
	vldr   d7, [r8, #8]
	vldr   d18, [r8, #16]   // k1
	vldr   d19, [r8, #24]
	add   r7,  r7,  r6,  lsl #2
	vmla.f32	q8,  q3 , d0[0]
	vmla.f32	q8,  q9 , d0[1]
	
	
	vldr   d0, [r7]
	vldr   d20, [r8, #48]   // k3
	vldr   d21, [r8, #56]
	vldr   d22, [r8, #64]   // k4
	vldr   d23, [r8, #72]
	sub  r7,  r7,  r6,  lsl #2
	add  r8,  r8, #144
	subs r4,  r4,  #1
	add  r7,  r7,  r10
	vmla.f32	q8,  q10, d0[0]
	vmla.f32	q8,  q11, d0[1]
	bne  loop_channel_9
	
activation_9:
	ldr	  r6, [sp, #52]
	cmp   r6, #0
	blt   save_result_9

	vmov.i64	q0, #0x0
	vmax.f32	q8 , q8 , q0

        beq   save_result_9

	vdup.32	        q0, r6
        vcvt.f32.s32    q0, q0
	vmin.f32	q8 , q8 , q0

save_result_9:

	vst1.32 d16[0], [r9], r10
	vst1.32 d16[1], [r9], r10
	vst1.32 d17[0], [r9], r10
	vst1.32 d17[1], [r9], r10

direct_k3s1p1_end:
	pop		{r4,r5,r6,r7,r8,r9,r10,r11,r12,lr}
	
	bx	lr
	
	.end


