/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license Agreements.  See the NOTICE file
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
// depthwise convolution kernel size 3x3 stride 1  pad 1
//
// input:
//         r0     arg0  input data address 
//         r1     arg1  kernel data address
//         r2     arg2  output data address
//         r3     arg3  channel number
//         sp     arg4  input width
//         sp+0x4 arg5  input height    must >=4
//         sp+0x8 arg6  bias point
// output: no
//
// register definition
//        r0         intput data address for every channel
//        r1         kernel address
//        r2         output data address for every channel
//        r3         channel counter
//        r4 sp+0x60 input width
//        r5         input  pointer
//        r6         output pointer
//        r7 sp+0x64 input height / line counter
//        r8         column counter
//        r10 sp+0x68 bias point 
// input  s0 s1 s2 s3 s4 s5
//
// kernel s7  s8  s9
//        s10 s11 s12
//        s13 s14 s15
//
// output s16  ~  s23
//        s24  ~  s31
#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1
#endif
	.section .text, "ax"
	.align 5
	.type KERNEL_NAME STT_FUNC
	.global KERNEL_NAME
	.hidden KERNEL_NAME
KERNEL_NAME:
	// context save & load parameter
	push		{r4 - r10, lr}
	vpush		{d8 - d15}
	ldr		r4,[sp, #0x60]	// r4 = input width 
        ldr             r10,[sp,#0x68]  // r10 = bias point
        #ifdef CONV_RELU_FUSE
	vmov.i64	q9, #0
        #ifdef CONV_RELU6_FUSE
        mov             r9, #0x6
        vdup.32         q10, r9
        vcvt.f32.s32    q10, q10
        #endif   	
	#endif
channel_loop:
        teq             r10, #0x0
        beq             no_biases
        vld1.f32        {d16[]},[r10]
        add             r10,r10, #0x4  // r10 = r10 + 4; 
        b               dw_start
no_biases:
        vmov.i64        d16, #0
dw_start:
        // load kernel
	vldm		r1!, {s7 - s15}
        // first 2 colunm	
	ldr		r7,[sp, #0x64]	// r7 initial line counter = input height - 2
	mov		r5, r0		// initial input  pointer
	mov		r6, r2		// initial output pointer
	sub		r7, r7, #2
	// first line
	vldr		d0, [r5]
	vmul.f32	s16, s0, s11
	vmul.f32	s24, s0, s8
	add		r5, r5, r4, LSL #2 
	vmla.f32	s16, s1, s12
	vmla.f32	s24, s1, s9
	// looped 2 more lines
first_column_line_loop:
	// line 1
	vldr		d0, [r5]
	add		r5, r5, r4, LSL #2 
	vmla.f32	s16, s0, s14
	vmla.f32	s24, s0, s11
	vldr		d1, [r5]
	vmla.f32	s16, s1, s15
	add		r5, r5, r4, LSL #2
	//add bias
        vadd.f32        d8,d8,d16
        #ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
        #ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
	vmla.f32	s24, s1, s12
	vstr		s16, [r6]
	vmul.f32	s16, s0, s8
	add		r6 , r6, r4, LSL #2
	vmla.f32	s24, s2, s14
	vmla.f32	s16, s1, s9
	vmla.f32	s24, s3, s15
	vmla.f32	s16, s2, s11
	//add bias
        vadd.f32        d12,d12,d16
        #ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12, d18
        #ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12, d20
	#endif
	#endif
	vstr		s24, [r6]
	vmul.f32	s24, s2, s8
	add		r6 , r6, r4, LSL #2
	vmla.f32	s16, s3, s12
	subs		r7 , r7, #2
	vmla.f32	s24, s3, s9
	bgt		first_column_line_loop
	tst		r7, #1			// if (height&0x1)
	bne		first_column_last_line
	vldr		d0, [r5]
	vmla.f32	s16, s0, s14
	vmla.f32	s24, s0, s11
	vmla.f32	s16, s1, s15
	vmla.f32	s24, s1, s12
        //add bias
        vadd.f32        d8, d8 ,d16
        vadd.f32        d12,d12,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d12,d12, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d12,d12, d20
        #endif
	#endif
	vstr		s16, [r6]
	add		r6 , r6, r4, LSL #2
	vstr		s24, [r6]
	b		first_column_end
first_column_last_line:
        // add bias
        vadd.f32        d8,d8,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8 ,d8 , d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8 ,d8 , d20
	#endif
	#endif
	vstr		s16, [r6]
first_column_end:
	mov		r8, #0			// r8 initial column counter = 0
    // looped 8 more columns
more8_column_loop:
	sub		r14,r4, #9
	cmp		r8, r14
	bge		more8_column_loop_end	// if less than 8 line left, break
	add		r5, r0, r8, LSL #2	// initial input  pointer
	add		r6, r2, r8, LSL #2	// initial output pointer
	vldm		r5,{s0 - s5}
	add		r6, r6, #4		//
	vmul.f32	s16, s0, s10
	pld		[r5, #0x28]
	vmul.f32	s17, s1, s10
        vmul.f32        d9,  d1, d5[0]
        vmul.f32        q6,  q0, d3[1]
	vmla.f32	s16, s1, s11
	vmla.f32	s17, s2, s11
	vmla.f32	s18, s3, s11
	vmla.f32	s19, s4, s11
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s26, s3, s8
	vmla.f32	s27, s4, s8
	vldr		d0, [r5, #0x18]
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
	vmla.f32	s18, s4, s12
	vmla.f32	s19, s5, s12
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
	vldr		d1, [r5, #0x20]
	add		r5, r5, r4, LSL #2 
	vmla.f32	s26, s4, s9
	vmla.f32	s27, s5, s9
	vmul.f32	s20, s4, s10
	vmul.f32	s21, s5, s10
	vmul.f32	s22, s0, s10
	vmul.f32	s23, s1, s10
	vmul.f32	s28, s4, s7
	vmul.f32	s29, s5, s7
	vmul.f32	s30, s0, s7
	vmul.f32	s31, s1, s7
	vmla.f32	s20, s5, s11
	vmla.f32	s21, s0, s11
	vmla.f32	s22, s1, s11
	vmla.f32	s23, s2, s11
	vmla.f32	s28, s5, s8
	vmla.f32	s29, s0, s8
	vmla.f32	s30, s1, s8
	vmla.f32	s31, s2, s8
	vmla.f32        q5,  q0, d6[0]
	ldr		r7,[sp, #0x64]		// r7 = line counter
	vmla.f32	s28, s0, s9
	vmla.f32	s29, s1, s9
	vmla.f32	s30, s2, s9
	vmla.f32	s31, s3, s9
	// looped 2 more lines
	sub		r7, r7, #2
more8_column_line_loop:
	  // line 1 left 4
	vldm		r5,{s0 - s5}
	pld		[r5, #0x28]
	vmla.f32	s16, s0, s13
	vmla.f32	s17, s1, s13
	vmla.f32	s18, s2, s13
	vmla.f32	s19, s3, s13
	vmla.f32        q6,  q0, d5[0]
	vmla.f32	s16, s1, s14
	vmla.f32	s17, s2, s14
	vmla.f32	s18, s3, s14
	vmla.f32	s19, s4, s14
	vmla.f32	s24, s1, s11
	vmla.f32	s25, s2, s11
	vmla.f32	s26, s3, s11
	vmla.f32	s27, s4, s11
	vmla.f32	s16, s2, s15
	vmla.f32	s17, s3, s15
	vmla.f32	s18, s4, s15
	vmla.f32	s19, s5, s15
	vmla.f32	s24, s2, s12
	vmla.f32	s25, s3, s12
        //add bias
        vadd.f32        d8,  d8, d16
        vadd.f32        d9,  d9, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8 , d18
	vmax.f32	d9, d9 , d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8 , d20
	vmin.f32	d9, d9 , d20
        #endif
	#endif
	vmla.f32	s26, s4, s12
	vmla.f32	s27, s5, s12
	vstm		r6, {d8,d9}
	vmul.f32	s16, s0, s7
	vmul.f32	s17, s1, s7
	vmul.f32	s18, s2, s7
	vmul.f32	s19, s3, s7
	vmla.f32	s16, s1, s8
	vmla.f32	s17, s2, s8
	vmla.f32	s18, s3, s8
	vmla.f32	s19, s4, s8
	vmla.f32	s16, s2, s9
	vmla.f32	s17, s3, s9
	vldr		d0, [r5, #0x18]
	vmla.f32	s18, s4, s9
	vmla.f32	s19, s5, s9
          // line 1 right 4
	vldr		d1, [r5, #0x20]
	vmla.f32	s20, s4, s13
	vmla.f32	s21, s5, s13
	vmla.f32	s22, s0, s13
	vmla.f32	s23, s1, s13
	vmla.f32	s28, s4, s10
	vmla.f32	s29, s5, s10
	vmla.f32	s30, s0, s10
	vmla.f32	s31, s1, s10
	vmla.f32	s20, s5, s14
	vmla.f32	s21, s0, s14
	vmla.f32	s22, s1, s14
	vmla.f32	s23, s2, s14
	vmla.f32	s28, s5, s11
	vmla.f32	s29, s0, s11
	vmla.f32	s30, s1, s11
	vmla.f32	s31, s2, s11
	vmla.f32	s20, s0, s15
	vmla.f32	s21, s1, s15
	vmla.f32	s22, s2, s15
	vmla.f32	s23, s3, s15
	vmla.f32	s28, s0, s12
	vmla.f32	s29, s1, s12
        //add bias
        vadd.f32        d10,d10,d16
        vadd.f32        d11,d11,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d10,d10,d18
	vmax.f32	d11,d11,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d10,d10,d20
	vmin.f32	d11,d11,d20
        #endif
	#endif
	vmla.f32	s30, s2, s12
	vmla.f32	s31, s3, s12
	add		r5, r5, r4, LSL #2 
	vstr		d10,[r6, #0x10]
	vstr		d11,[r6, #0x18]
	add		r6, r6, r4, LSL #2 
	vmul.f32	s20, s4, s7
	vmul.f32	s21, s5, s7
	vmul.f32	s22, s0, s7
	vmul.f32	s23, s1, s7
	vmla.f32	s20, s5, s8
	vmla.f32	s21, s0, s8
	vmla.f32	s22, s1, s8
	vmla.f32	s23, s2, s8
	vmla.f32	s20, s0, s9
	vmla.f32	s21, s1, s9
	vmla.f32	s22, s2, s9
	vmla.f32	s23, s3, s9
	  // line 2 left 4
	vldm		r5,{s0 - s5}
	pld		[r5, #0x28]
	vmla.f32	s24, s0, s13
	vmla.f32	s25, s1, s13
	vmla.f32	s26, s2, s13
	vmla.f32	s27, s3, s13
	vmla.f32	s16, s0, s10
	vmla.f32	s17, s1, s10
	vmla.f32	s18, s2, s10
	vmla.f32	s19, s3, s10
	vmla.f32	s24, s1, s14
	vmla.f32	s25, s2, s14
	vmla.f32	s26, s3, s14
	vmla.f32	s27, s4, s14
	vmla.f32	s16, s1, s11
	vmla.f32	s17, s2, s11
	vmla.f32	s18, s3, s11
	vmla.f32	s19, s4, s11
	vmla.f32	s24, s2, s15
	vmla.f32	s25, s3, s15
	vmla.f32	s26, s4, s15
	vmla.f32	s27, s5, s15
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
        //add bias
        vadd.f32        d12,d12,d16
        vadd.f32        d13,d13,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12,d18
	vmax.f32	d13,d13,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12,d20
	vmin.f32	d13,d13,d20
        #endif
	#endif
	vmla.f32	s18, s4, s12
	vmla.f32	s19, s5, s12
	vstm		r6, {d12,d13}
	vmul.f32	s24, s0, s7
	vmul.f32	s25, s1, s7
	vmul.f32	s26, s2, s7
	vmul.f32	s27, s3, s7
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s26, s3, s8
	vmla.f32	s27, s4, s8
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
	vldr		d0, [r5, #0x18]
	vmla.f32	s26, s4, s9
	vmla.f32	s27, s5, s9
	  // line 2 right 4
	vldr		d1, [r5, #0x20]
	vmla.f32	s28, s4, s13
	vmla.f32	s29, s5, s13
	vmla.f32	s30, s0, s13
	vmla.f32	s31, s1, s13
	vmla.f32	s20, s4, s10
	vmla.f32	s21, s5, s10
	vmla.f32	s22, s0, s10
	vmla.f32	s23, s1, s10
	vmla.f32	s28, s5, s14
	vmla.f32	s29, s0, s14
	vmla.f32	s30, s1, s14
	vmla.f32	s31, s2, s14
	vmla.f32	s20, s5, s11
	vmla.f32	s21, s0, s11
	vmla.f32	s22, s1, s11
	vmla.f32	s23, s2, s11
	vmla.f32	s28, s0, s15
	vmla.f32	s29, s1, s15
	vmla.f32	s30, s2, s15
	vmla.f32	s31, s3, s15
	vmla.f32	s20, s0, s12
 	vmla.f32	s21, s1, s12
        //add bias
        vadd.f32        d14,d14,d16
        vadd.f32        d15,d15,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14,d14,d18
	vmax.f32	d15,d15,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14,d14,d20
	vmin.f32	d15,d15,d20
        #endif
	#endif
	vmla.f32	s22, s2, s12
	vmla.f32	s23, s3, s12
	add		r5, r5, r4, LSL #2 
	vstr		d14, [r6, #0x10]
	vstr		d15, [r6, #0x18]
	add		r6, r6, r4, LSL #2 
	vmul.f32	s28, s4, s7
	vmul.f32	s29, s5, s7
	vmul.f32	s30, s0, s7
	vmul.f32	s31, s1, s7
	vmla.f32	s28, s5, s8
	vmla.f32	s29, s0, s8
	vmla.f32	s30, s1, s8
	vmla.f32	s31, s2, s8
	vmla.f32	s28, s0, s9
	vmla.f32	s29, s1, s9
	vmla.f32	s30, s2, s9
	vmla.f32	s31, s3, s9
	subs		r7 , r7, #2
	bgt		more8_column_line_loop
	tst		r7, #1			// if(height&0x1)
	bne		more8_column_last_line
	// 8 more column last 1 line
	  // last line left 4
	vldm		r5,{s0 - s5}
	pld		[r5, #0x28]
	vmla.f32	s16, s0, s13
	vmla.f32	s17, s1, s13
	vmla.f32	s18, s2, s13
	vmla.f32	s19, s3, s13
	vmla.f32	s24, s0, s10
	vmla.f32	s25, s1, s10
	vmla.f32	s26, s2, s10
	vmla.f32	s27, s3, s10
	vmla.f32	s16, s1, s14
	vmla.f32	s17, s2, s14
	vmla.f32	s18, s3, s14
	vmla.f32	s19, s4, s14
	vmla.f32	s24, s1, s11
	vmla.f32	s25, s2, s11
	vmla.f32	s26, s3, s11
	vmla.f32	s27, s4, s11
	vmla.f32	s16, s2, s15
	vmla.f32	s17, s3, s15
	vmla.f32	s18, s4, s15
	vmla.f32	s19, s5, s15
	vmla.f32	s24, s2, s12
	vmla.f32	s25, s3, s12
        //add bias
        vadd.f32       d8,d8,d16
        vadd.f32       d9,d9,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d9, d9, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d9, d9, d20
        #endif
	#endif
	vmla.f32	s26, s4, s12
	vmla.f32	s27, s5, s12
	vstm		r6, {d8,d9}
	  // last line right 4
	vldr		d0, [r5, #0x18]
	vldr		d1, [r5, #0x20]
	vmla.f32	s20, s4, s13
	vmla.f32	s21, s5, s13
	vmla.f32	s22, s0, s13
	vmla.f32	s23, s1, s13
	vmla.f32	s28, s4, s10
	vmla.f32	s29, s5, s10
	vmla.f32	s30, s0, s10
	vmla.f32	s31, s1, s10
	vmla.f32	s20, s5, s14
	vmla.f32	s21, s0, s14
	vmla.f32	s22, s1, s14
	vmla.f32	s23, s2, s14
	vmla.f32	s28, s5, s11
	vmla.f32	s29, s0, s11
	vmla.f32	s30, s1, s11
	vmla.f32	s31, s2, s11
	vmla.f32	s20, s0, s15
	vmla.f32	s21, s1, s15
	vmla.f32	s22, s2, s15
	vmla.f32	s23, s3, s15
	vmla.f32	s28, s0, s12
	vmla.f32	s29, s1, s12
	vmla.f32	s30, s2, s12
	vmla.f32	s31, s3, s12
        //add bias
        vadd.f32        d10,d10,d16
        vadd.f32        d11,d11,d16
        vadd.f32        d12,d12,d16
        vadd.f32        d13,d13,d16
        vadd.f32        d14,d14,d16
        vadd.f32        d15,d15,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q5, q5, q9
	vmax.f32	q6, q6, q9
	vmax.f32	q7, q7, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q5, q5, q10
	vmin.f32	q6, q6, q10
	vmin.f32	q7, q7, q10
        #endif
	#endif
	vstr		d10,[r6, #0x10]
	vstr		d11,[r6, #0x18]
	add		r6, r6, r4, LSL #2
	vstm		r6,{d12-d15}
	b 		more8_column_line_loop_end
more8_column_last_line:
        //add bias
        vadd.f32       d8,d8,d16
        vadd.f32       d9,d9,d16
        vadd.f32       d10,d10,d16
        vadd.f32       d11,d11,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q4, q4, q9
	vmax.f32	q5, q5, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q4, q4, q10
	vmin.f32	q5, q5, q10
        #endif
	#endif
	vstm		r6,{d8-d11}
more8_column_line_loop_end:
	add		r8, r8, #8
	b		more8_column_loop
more8_column_loop_end:
    // looped 4 more columns
	sub		r14,r4, #5
	cmp		r8, r14
	bge		more4_column_loop_end	// if less than 4 line left, break
	add		r5, r0, r8, LSL #2	// initial input  pointer
	add		r6, r2, r8, LSL #2	// initial output pointer
	add		r6, r6, #4		//
	vldm		r5,{s0 - s5}
	vmul.f32	s16, s0, s10
	vmul.f32	s17, s1, s10
	vmul.f32	s18, s2, s10
	vmul.f32	s19, s3, s10
	pld		[r5, #0x18]
	vmul.f32	s24, s0, s7
	vmul.f32	s25, s1, s7
	vmul.f32	s26, s2, s7
	vmul.f32	s27, s3, s7
	vmla.f32	s16, s1, s11
	add		r5, r5, r4, LSL #2 
	vmla.f32	s17, s2, s11
	vmla.f32	s18, s3, s11
	vmla.f32	s19, s4, s11
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s26, s3, s8
	vmla.f32	s27, s4, s8
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
	vmla.f32	s18, s4, s12
	vmla.f32	s19, s5, s12
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
	vmla.f32	s26, s4, s9
	vmla.f32	s27, s5, s9
	// looped 2 more lines
	ldr		r7,[sp, #0x64]		// r7 = line counter
	sub		r7, r7, #2
more4_column_line_loop:
	// line 1
	vldm		r5,{s0 - s5}
	vmla.f32	s16, s0, s13
	vmla.f32	s17, s1, s13
	vmla.f32	s18, s2, s13
	vmla.f32	s19, s3, s13
	pld		[r5, #0x18]
	vmla.f32	s24, s0, s10
	vmla.f32	s25, s1, s10
	vmla.f32	s26, s2, s10
	vmla.f32	s27, s3, s10
	vmla.f32	s16, s1, s14
	vmla.f32	s17, s2, s14
	vmla.f32	s18, s3, s14
	vmla.f32	s19, s4, s14
	vmla.f32	s24, s1, s11
	vmla.f32	s25, s2, s11
	vmla.f32	s26, s3, s11
	vmla.f32	s27, s4, s11
	vmla.f32	s16, s2, s15
	vmla.f32	s17, s3, s15
	vmla.f32	s18, s4, s15
	vmla.f32	s19, s5, s15
	vmla.f32	s24, s2, s12
	vmla.f32	s25, s3, s12
        //add bias
        vadd.f32        d8,d8,d16
        vadd.f32        d9,d9,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d9, d9, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d9, d9, d20
        #endif
	#endif
	vmla.f32	s26, s4, s12
	vmla.f32	s27, s5, s12
	add		r5, r5, r4, LSL #2 
	vstm		r6, {d8,d9}
	add		r6, r6, r4, LSL #2 
	vmul.f32	s16, s0, s7
	vmul.f32	s17, s1, s7
	vmul.f32	s18, s2, s7
	vmul.f32	s19, s3, s7
	vmla.f32	s16, s1, s8
	vmla.f32	s17, s2, s8
	vmla.f32	s18, s3, s8
	vmla.f32	s19, s4, s8
	vmla.f32	s16, s2, s9
	vmla.f32	s17, s3, s9
	vmla.f32	s18, s4, s9
	vmla.f32	s19, s5, s9
	// line 2
	vldm		r5,{s0 - s5}
	vmla.f32	s24, s0, s13
	vmla.f32	s25, s1, s13
	vmla.f32	s26, s2, s13
	vmla.f32	s27, s3, s13
	vmla.f32	s16, s0, s10
	vmla.f32	s17, s1, s10
	vmla.f32	s18, s2, s10
	vmla.f32	s19, s3, s10
	pld		[r5, #0x18]
	vmla.f32	s24, s1, s14
	vmla.f32	s25, s2, s14
	vmla.f32	s26, s3, s14
	vmla.f32	s27, s4, s14
	vmla.f32	s16, s1, s11
	vmla.f32	s17, s2, s11
	vmla.f32	s18, s3, s11
	vmla.f32	s19, s4, s11
	vmla.f32	s24, s2, s15
	vmla.f32	s25, s3, s15
	vmla.f32	s26, s4, s15
	vmla.f32	s27, s5, s15
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
        //add bias
        vadd.f32        d12, d12,d16
        vadd.f32        d13, d13,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12, d18
	vmax.f32	d13,d13, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12, d20
	vmin.f32	d13,d13, d20
        #endif
	#endif
	vmla.f32	s18, s4, s12
	vmla.f32	s19, s5, s12
	add		r5, r5, r4, LSL #2 
	vstm		r6, {d12,d13}
	add		r6, r6, r4, LSL #2 
	vmul.f32	s24, s0, s7
	vmul.f32	s25, s1, s7
	vmul.f32	s26, s2, s7
	vmul.f32	s27, s3, s7
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s26, s3, s8
	vmla.f32	s27, s4, s8
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
	vmla.f32	s26, s4, s9
	vmla.f32	s27, s5, s9
	subs		r7 , r7, #2
	bgt		more4_column_line_loop
	tst		r7, #1			// if(height&0x1)
	bne		more4_column_last_line
	// 4 more column last 1 line
	  // last line
	vldm		r5,{s0 - s5}
	vmla.f32	s16, s0, s13
	vmla.f32	s17, s1, s13
	vmla.f32	s18, s2, s13
	vmla.f32	s19, s3, s13
	vmla.f32	s24, s0, s10
	vmla.f32	s25, s1, s10
	vmla.f32	s26, s2, s10
	vmla.f32	s27, s3, s10
	pld		[r5, #0x18]
	vmla.f32	s16, s1, s14
	vmla.f32	s17, s2, s14
	vmla.f32	s18, s3, s14
	vmla.f32	s19, s4, s14
	add		r5, r5, r4, LSL #2 
	vmla.f32	s24, s1, s11
	vmla.f32	s25, s2, s11
	vmla.f32	s26, s3, s11
	vmla.f32	s27, s4, s11
	vmla.f32	s16, s2, s15
	vmla.f32	s17, s3, s15
	vmla.f32	s18, s4, s15
	vmla.f32	s19, s5, s15
	vmla.f32	s24, s2, s12
	vmla.f32	s25, s3, s12
	vmla.f32	s26, s4, s12
	vmla.f32	s27, s5, s12
        //add bias
        vadd.f32        d8,  d8, d16
        vadd.f32        d9,  d9, d16
        vadd.f32        d12, d12,d16
        vadd.f32        d13, d13,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q4, q4, q9
	vmax.f32	q6, q6, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q4, q4, q10
	vmin.f32	q6, q6, q10
        #endif
	#endif
	vstm		r6, {d8  - d9}
	add		r6, r6, r4, LSL #2
	vstm		r6, {d12 - d13}
	b 		more4_column_line_loop_end
more4_column_last_line:
        //add bias
        vadd.f32       d8,d8,d16
        vadd.f32       d9,d9,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q4, q4, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q4, q4, q10
	#endif
	#endif
	vstm		r6, {d8-d9}
more4_column_line_loop_end:
	add		r8, r8, #4
more4_column_loop_end:
	pld		[r1]
    // looped 2 more columns
	sub		r14,r4, #3
	cmp		r8, r14
	bge		more2_column_loop_end	// if less than 3 line left, break
	add		r5, r0, r8, LSL #2	// initial input pointer
	add		r6, r2, r8, LSL #2	// initial output pointer 
	add		r6, r6, #4		//
	// first line
	vldm		r5,{s0 - s3}
	add		r5, r5, r4, LSL #2 
	vmul.f32	s16, s0, s10
	vmul.f32	s17, s1, s10
	vmul.f32	s24, s0, s7
	vmul.f32	s25, s1, s7
	vmla.f32	s16, s1, s11
	vmla.f32	s17, s2, s11
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
	// looped 2 more lines
	ldr		r7,[sp, #0x64]		// r7 = line counter
	sub		r7, r7, #2
more2_column_line_loop:
	// line 1
	vldm		r5,{s0 - s3}
	add		r5, r5, r4, LSL #2 
	vmla.f32	s16, s0, s13
	vmla.f32	s17, s1, s13
	vmla.f32	s24, s0, s10
	vmla.f32	s25, s1, s10
	vmla.f32	s16, s1, s14
	vmla.f32	s17, s2, s14
	vmla.f32	s24, s1, s11
	vmla.f32	s25, s2, s11
	vmla.f32	s16, s2, s15
	vmla.f32	s17, s3, s15
        //add bias
        vadd.f32        d8 , d8, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
	vmla.f32	s24, s2, s12
	vmla.f32	s25, s3, s12
	vstr		d8, [r6]
	add		r6, r6, r4, LSL #2 
	vmul.f32	s16, s0, s7
	vmul.f32	s17, s1, s7
	vmla.f32	s16, s1, s8
	vmla.f32	s17, s2, s8
	vmla.f32	s16, s2, s9
	vmla.f32	s17, s3, s9
	// line 2
	vldm		r5,{s0 - s3}
	add		r5, r5, r4, LSL #2 
	vmla.f32	s24, s0, s13
	vmla.f32	s25, s1, s13
	vmla.f32	s16, s0, s10
	vmla.f32	s17, s1, s10
	vmla.f32	s24, s1, s14
	vmla.f32	s25, s2, s14
	vmla.f32	s16, s1, s11
	vmla.f32	s17, s2, s11
	vmla.f32	s24, s2, s15
	vmla.f32	s25, s3, s15
        //add bias
        vadd.f32        d12,d12,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12, d20
	#endif
	#endif
	vmla.f32	s16, s2, s12
	vmla.f32	s17, s3, s12
	vstr		d12, [r6]
	add		r6, r6, r4, LSL #2 
	vmul.f32	s24, s0, s7
	vmul.f32	s25, s1, s7
	vmla.f32	s24, s1, s8
	vmla.f32	s25, s2, s8
	vmla.f32	s24, s2, s9
	vmla.f32	s25, s3, s9
        subs            r7 , r7, #2
        bgt             more2_column_line_loop
        tst             r7, #1			// if(height&0x1)
        bne             more2_column_last_line
        // more 2 column last 1 line
        vldm            r5,{s0 - s3}
        vmla.f32        s16, s0, s13
        vmla.f32        s17, s1, s13
        vmla.f32        s24, s0, s10
        vmla.f32        s25, s1, s10
        vmla.f32        s16, s1, s14
        vmla.f32        s17, s2, s14
        vmla.f32        s24, s1, s11
        vmla.f32        s25, s2, s11
        vmla.f32        s16, s2, s15
        vmla.f32        s17, s3, s15
        vmla.f32        s24, s2, s12
        vmla.f32        s25, s3, s12
        //add bias
        vadd.f32        d8, d8, d16
        vadd.f32        d12,d12,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d12,d12,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d12,d12,d20
        #endif
	#endif
        vstr            d8, [r6]
        add             r6, r6, r4, LSL #2 
        vstr            d12,[r6]
        b               more2_column_line_loop_end
more2_column_last_line:
        //add bias
        vadd.f32        d8, d8, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
        vstr            d8, [r6]
more2_column_line_loop_end:
        add             r8, r8, #2
more2_column_loop_end:
	ldr		r14,[sp, #0x64]		// input height
	mul		r14, r14, r4		// input width * height
    // last 1 column
	tst		r4, #1		// if(width&0x1)
	beq		last_0_column
        add             r5, r0, r8, LSL #2      // initial input pointer 
        add             r6, r2, r8, LSL #2      // initial output pointer
	add		r6, r6, #4
        // first line
        vldm            r5,{s0 - s2}
	pld		[r5, r14, LSL #2]
        vmul.f32        s16, s0, s10
        vmul.f32        s17, s1, s10
        add             r5, r5, r4, LSL #2 
        vmul.f32        s24, s0, s7
        vmul.f32        s25, s1, s7
        vmla.f32        s16, s1, s11
        vmla.f32        s17, s2, s11
        vmla.f32        s24, s1, s8
        vmla.f32        s25, s2, s8
        vmla.f32        s16, s2, s12
        vmla.f32        s24, s2, s9
        // looped 2 more lines
        ldr             r7,[sp, #0x64]          // r7 = line counter
        sub             r7, r7, #2
last1_column_line_loop:
        // line 1
        vldm            r5,{s0 - s2}
        vmla.f32        s16, s0, s13
        vmla.f32        s17, s1, s13
	pld		[r5, r14, LSL #2]
        vmla.f32        s24, s0, s10
        vmla.f32        s25, s1, s10
        vmla.f32        s16, s1, s14
        add             r5, r5, r4, LSL #2 
        vmla.f32        s17, s2, s14
        vmla.f32        s24, s1, s11
        vmla.f32        s25, s2, s11
        vmla.f32        s16, s2, s15
        vmla.f32        s24, s2, s12
        //add bias
        vadd.f32        d8, d8, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
        vstr            d8, [r6]
        add             r6, r6, r4, LSL #2 
        vmul.f32        s16, s0, s7
        vmul.f32        s17, s1, s7
        vmla.f32        s16, s1, s8
        vmla.f32        s17, s2, s8
        vmla.f32        s16, s2, s9
        // line 2
        vldm            r5,{s0 - s2}
        vmla.f32        s24, s0, s13
        vmla.f32        s25, s1, s13
	pld		[r5, r14, LSL #2]
        vmla.f32        s16, s0, s10
        vmla.f32        s17, s1, s10
        add             r5, r5, r4, LSL #2 
        vmla.f32        s24, s1, s14
        vmla.f32        s25, s2, s14
        vmla.f32        s16, s1, s11
        vmla.f32        s17, s2, s11
        vmla.f32        s24, s2, s15
        vmla.f32        s16, s2, s12
        //add bias
        vadd.f32        d12, d12, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12, d20
	#endif
	#endif
        vstr            d12, [r6]
        add             r6, r6, r4, LSL #2 
        vmul.f32        s24, s0, s7
        vmul.f32        s25, s1, s7
        vmla.f32        s24, s1, s8
        vmla.f32        s25, s2, s8
        vmla.f32        s24, s2, s9
        subs            r7 , r7, #2
        bgt             last1_column_line_loop
        tst             r7, #1			// if(height*0x1)
        bne             last1_column_last_line
        // last column last 1 line
        vldm            r5,{s0 - s2}
        vmla.f32        s16, s0, s13
        vmla.f32        s17, s1, s13
	pld		[r5, r14, LSL #2]
        vmla.f32        s24, s0, s10
        vmla.f32        s25, s1, s10
        add             r5, r5, r4, LSL #2 
        vmla.f32        s16, s1, s14
        vmla.f32        s17, s2, s14
        vmla.f32        s24, s1, s11
        vmla.f32        s25, s2, s11
        vmla.f32        s16, s2, s15
        vmla.f32        s24, s2, s12
        //add bias
        vadd.f32        d8,d8,d16
        vadd.f32        d12,d12,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d12,d12,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d12,d12,d20
        #endif
	#endif
        vstr            d8, [r6]
        add             r6, r6, r4, LSL #2 
        vstr            d12,[r6]
        b               next_channel
last1_column_last_line:
        //add bias
        vadd.f32        d8,d8,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
        vstr            d8, [r6]
	b		next_channel
    // last 0 column
last_0_column:	
        add             r5, r0, r8, LSL #2      // initial input  pointer
        add             r6, r2, r8, LSL #2      // initial output pointer
	add		r6, r6, #4
        // first line
        vldm            r5,{s0 - s1}
        add             r5, r5, r4, LSL #2 
        vmul.f32        s16, s0, s10
        vmul.f32        s24, s0, s7
        vmla.f32        s16, s1, s11
        vmla.f32        s24, s1, s8
        // looped 2 more lines
        ldr             r7,[sp, #0x64]          // r7 = line counter
        sub             r7, r7, #2
last0_column_line_loop:
        // line 1
        vldm            r5,{s0 - s1}
        vmla.f32        s16, s0, s13
        vmla.f32        s24, s0, s10
	pld		[r5, r14, LSL #2]
        vmla.f32        s16, s1, s14
        add             r5, r5, r4, LSL #2 
        //add bias
        vadd.f32        d8,d8,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	#endif
	#endif
        vmla.f32        s24, s1, s11
        vstr            s16, [r6]
        add             r6, r6, r4, LSL #2 
        vmul.f32        s16, s0, s7
        vmla.f32        s16, s1, s8
        // line 2
        vldm            r5,{s0 - s1}
        vmla.f32        s24, s0, s13
        vmla.f32        s16, s0, s10
	pld		[r5, r14, LSL #2]
        vmla.f32        s24, s1, s14
        add             r5, r5, r4, LSL #2 
        //add bias
        vadd.f32        d12,d12, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d12,d12, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d12,d12, d20
	#endif
	#endif
        vmla.f32        s16, s1, s11
        vstr            s24, [r6]
        add             r6, r6, r4, LSL #2 
        vmul.f32        s24, s0, s7
        vmla.f32        s24, s1, s8
        subs            r7 , r7, #2
        bgt             last0_column_line_loop
        tst             r7, #1
        bne             last0_column_last_line
        // last column last 1 line
        vldm            r5,{s0 - s1}
        vmla.f32        s16, s0, s13
        vmla.f32        s24, s0, s10
	pld		[r5, r14, LSL #2]
        vmla.f32        s16, s1, s14
        vmla.f32        s24, s1, s11
        //add bias
        vadd.f32        d8, d8,  d16
        vadd.f32        d12,d12, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d8, d8, d18
	vmax.f32	d12,d12,d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d8, d8, d20
	vmin.f32	d12,d12,d20
        #endif
	#endif
        vstr            s16, [r6]
        add             r6, r6, r4, LSL #2 
        vstr            s24,[r6]
        b               next_channel
last0_column_last_line:
        vstr            s16, [r6]
next_channel:
	// set next channel input output address
	add		r0, r0, r14, LSL #2	// new input  address
	add		r2, r2, r14, LSL #2	// new output address
	subs		r3, r3, #1
	bne		channel_loop
	// restore content
	vpop		{d8 - d15}
	pop		{r4 - r10, pc}
	.end
