/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file Except in compliance
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
// depthwise convolution kernel size 3x3 stride 2
//
// input:
//         r0     arg0  input data address 
//         r1     arg1  kernel data address
//         r2     arg2  output data address
//         r3     arg3  channel number
//         sp     arg4  input width
//         sp+0x4 arg5  input height
//         sp+0x8 arg6  bias  data
//         sp+0xc arg7  left pad 1 flag
// output: no
//
// register definition
//        r0         intput data address for every channel
//        r1         kernel address
//        r2         output data address for every channel
//        r3         channel counter
//        r4 sp+0x68 input width
//        r5         output width
//        r6         input line
//        r7         output line
//        r8 sp+0x6c input height / line counter
//        r9         column counter
//        r10 sp+0x70 bias data address 
//        r11 sp+0x74 left pad 1 flag
//        r14         tmp register
// input  s0  ~  s8
//        s9  ~  s17
//
// kernel s18 s19 s20
//        s21 s22 s23
//        s24 s25 s26
//
// output s28 s29 s30 s31
//
// bias   d16
// relu   d18 d19
//
#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2
#endif
	.section .text, "ax"
	.align 5
	.type KERNEL_NAME STT_FUNC
	.global KERNEL_NAME
	.hidden KERNEL_NAME
KERNEL_NAME:
	// context save & load parameter
	push		{r4 - r12, lr}
	vpush		{d8 - d15}
	ldr		r4,[sp, #0x68]		// r4 = input width
	ldr		r11,[sp,#0x74]  	// r11= left pad 1 flag 
	add		r5, r4, r11
	lsr		r5, r5, #1		// r5 = output width
        ldr             r10,[sp,#0x70]		// bias data address
	#ifdef CONV_RELU_FUSE
	vmov.i64	d18, #0			// for relu
	vmov.i64	d19, #0			// for relu
	#ifdef CONV_RELU6_FUSE
        mov             r12, #0x6
        vdup.32         q10, r12
        vcvt.f32.s32    q10, q10            
        #endif
	#endif
channel_loop:
        teq             r10 ,#0x0
        beq             no_biases
        vld1.f32        {d16[]},[r10]   	// load bias
        add             r10,r10,#0x4		// r10 = r10 + sizeof(float)
        b               dw_start
no_biases:
       vmov.i64         d16, #0
dw_start:
   // load kernel
	vldm		r1!, {s18 - s26}
	mov		r9, #0			// r9 initial column counter = 0
	mov		r6, r0			// initial input  line address
	mov		r7, r2			// initial output line address

	cmp		r11, #0			// if left pad = 0, skip first column
	beq		more8_column_loop

   // first 2 colunm	
	vmov.i64	d14, #0			// padding value
	ldr		r8,[sp, #0x6c]		// r8 initial line counter = input height - 1
	sub		r8, r8, #1

	cmp		r11, #0			// if pad=1, first line is 0
	bne		first_column_line_loop	// if pad = 0, accumulate first line
	// accumulate first line    will this branch be executed?
	vldr		d0, [r6]
	pld		[r6, #0x8]
	add		r6, r6, r4, LSL #2 
	vmla.f32	s28, s0, s19
	sub		r8, r8, #1
	vmla.f32	s28, s1, s20	

	// looped 2 more lines
first_column_line_loop:
	vldr		d0, [r6]
	pld		[r6, #0x8]
	add		r6, r6, r4, LSL #2 
	vmla.f32	s28, s0, s22
	vldm		r6,{s9, s10}
	pld		[r6, #0x8]
	vmla.f32	s28, s1, s23
	add		r6, r6, r4, LSL #2
	vmla.f32	s28, s9, s25
	vmla.f32	s28, s10,s26
        //add bias
        vadd.f32        d14, d14, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
	vmul.f32	s28, s9, s19
	add		r7 , r7, r5, LSL #2
	subs		r8 , r8, #2
	vmla.f32	s28, s10,s20
	bgt		first_column_line_loop
	tst		r8, #1
	bne		first_column_end
	// last line
	vldr		d0, [r6]
	vmla.f32	s28, s0, s22
	vmla.f32	s28, s1, s23
        //add bias
        vadd.f32        d14,d14, d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
first_column_end:
	add		r9, r9, #1
    // looped 8 more column
more8_column_loop:
	sub		r14,r4, #8
	cmp		r9, r14
	bge		more8_column_loop_end	// if less than 8 line left, break
	add		r6, r0, r9, LSL #2	// initial input line address
	add		r14,r9, r11
	add		r7, r2, r14,LSL #1	// initial output line address
	vmov.i64	d14,#0x0		// padding value
	vmov.i64	d15,#0x0		//
	// looped 2 more lines
	ldr		r8,[sp, #0x6c]		// r8 = line counter
	sub		r8, r8, #1

	cmp		r11, #0			// if pad=1, first line is 0
	bne		more8_column_line_loop	// if pad=0, accumulate first line
	// accumulate first line
	vldm		r6,{s0 - s8}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2 
	sub		r8, r8, #1
	vmla.f32	s28, s0, s18
	vmla.f32	s29, s2, s18
	vmla.f32	s30, s4, s18
	vmla.f32	s31, s6, s18
	vmla.f32	s28, s1, s19
	vmla.f32	s29, s3, s19
	vmla.f32	s30, s5, s19
	vmla.f32	s31, s7, s19
	vmla.f32	s28, s2, s20
	vmla.f32	s29, s4, s20
	vmla.f32	s30, s6, s20
	vmla.f32	s31, s8, s20

more8_column_line_loop:
	vldm		r6,{s0 - s8}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2 
	vldm		r6,{s9 - s17}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2
	vmla.f32	s28, s0, s21
	vmla.f32	s29, s2, s21
	vmla.f32	s30, s4, s21
	vmla.f32	s31, s6, s21
	vmla.f32	s28, s1, s22
	vmla.f32	s29, s3, s22
	vmla.f32	s30, s5, s22
	vmla.f32	s31, s7, s22
	vmla.f32	s28, s2, s23
	vmla.f32	s29, s4, s23
	vmla.f32	s30, s6, s23
	vmla.f32	s31, s8, s23
	vmla.f32	s28, s9, s24
	vmla.f32	s29, s11,s24
	vmla.f32	s30, s13,s24
	vmla.f32	s31, s15,s24
	vmla.f32	s28, s10,s25
	vmla.f32	s29, s12,s25
	vmla.f32	s30, s14,s25
	vmla.f32	s31, s16,s25
	vmla.f32	s28, s11,s26
	vmla.f32	s29, s13,s26
	vmla.f32	s30, s15,s26
	vmla.f32	s31, s17,s26
        //add bias
        vadd.f32        d14, d14,d16
        vadd.f32        d15, d15,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q7,  q7, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q7,  q7, q10
	#endif
	#endif
	vstm		r7, {d14, d15}
	add		r7, r7, r5, LSL #2
	vmul.f32	s28, s9, s18
	vmul.f32	s29, s11,s18
	vmul.f32	s30, s13,s18
	vmul.f32	s31, s15,s18
	vmla.f32	s28, s10,s19
	vmla.f32	s29, s12,s19
	vmla.f32	s30, s14,s19
	vmla.f32	s31, s16,s19
	vmla.f32	s28, s11,s20
	vmla.f32	s29, s13,s20
	vmla.f32	s30, s15,s20
	vmla.f32	s31, s17,s20
	subs		r8 , r8, #2
	bgt		more8_column_line_loop
	tst		r8, #1
	bne		more8_column_line_loop_end
	// last line
	vldm		r6,{s0 - s8}
	vmla.f32	s28, s0, s21
	vmla.f32	s29, s2, s21
	vmla.f32	s30, s4, s21
	vmla.f32	s31, s6, s21
	vmla.f32	s28, s1, s22
	vmla.f32	s29, s3, s22
	vmla.f32	s30, s5, s22
	vmla.f32	s31, s7, s22
	vmla.f32	s28, s2, s23
	vmla.f32	s29, s4, s23
	vmla.f32	s30, s6, s23
	vmla.f32	s31, s8, s23
        //add bias
        vadd.f32        d14,d14,d16
        vadd.f32        d15,d15,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	q7, q7, q9
	#ifdef CONV_RELU6_FUSE
	vmin.f32	q7, q7, q10
	#endif
	#endif
	vstm		r7, {d14, d15}
more8_column_line_loop_end:
	add		r9, r9, #8
	b		more8_column_loop
more8_column_loop_end:
    // 4 more columns
	sub		r14,r4, #4
	cmp		r9, r14
	bge		more4_column_end	// if less than 4 line left, skip
	add		r6, r0, r9, LSL #2	// initial input line address
	add		r14,r9, r11
	add		r7, r2, r14,LSL #1	// initial output line address
	vmov.i64	d14,#0x0		// padding value
	// looped 2 more lines
	ldr		r8,[sp, #0x6c]		// r8 = line counter
	sub		r8, r8, #1

	cmp		r11, #0			// if pad=1, first line is 0
	bne		more4_column_line_loop	// if pad=0, accumulate first line
	// accumulate first line
	vldm		r6,{s0 - s4}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2 
	sub		r8, r8, #1
	vmla.f32	s28, s0, s18
	vmla.f32	s29, s2, s18
	vmla.f32	s28, s1, s19
	vmla.f32	s29, s3, s19
	vmla.f32	s28, s2, s20
	vmla.f32	s29, s4, s20

more4_column_line_loop:
	vldm		r6,{s0 - s4}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2 
	vldm		r6,{s9 - s13}
	pld		[r6, #0x44]
	add		r6, r6, r4, LSL #2
	vmla.f32	s28, s0, s21
	vmla.f32	s29, s2, s21
	vmla.f32	s28, s1, s22
	vmla.f32	s29, s3, s22
	vmla.f32	s28, s2, s23
	vmla.f32	s29, s4, s23
	vmla.f32	s28, s9, s24
	vmla.f32	s29, s11,s24
	vmla.f32	s28, s10,s25
	vmla.f32	s29, s12,s25
	vmla.f32	s28, s11,s26
	vmla.f32	s29, s13,s26
        //add bias
        vadd.f32        d14,d14,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		d14, [r7]
	add		r7, r7, r5, LSL #2
	vmul.f32	s28, s9, s18
	vmul.f32	s29, s11,s18
	vmla.f32	s28, s10,s19
	vmla.f32	s29, s12,s19
	vmla.f32	s28, s11,s20
	vmla.f32	s29, s13,s20
	subs		r8 , r8, #2
	bgt		more4_column_line_loop
	tst		r8, #1
	bne		more4_column_line_loop_end
	// last line
	vldm		r6,{s0 - s4}
	vmla.f32	s28, s0, s21
	vmla.f32	s29, s2, s21
	vmla.f32	s28, s1, s22
	vmla.f32	s29, s3, s22
	vmla.f32	s28, s2, s23
	vmla.f32	s29, s4, s23
        //add bias	
        vadd.f32        d14,d14,d16
        #ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
        #ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		d14, [r7]
more4_column_line_loop_end:
	add		r9, r9, #4
more4_column_end:
	pld		[r1]
    // 2 more columns
	sub		r14, r4, #2
	cmp		r9, r14
	bge		more2_column_end	// if less than 2 line left, skip
	add		r6, r0, r9, LSL #2	// initial input line address
	add		r14,r9, r11
	add		r7, r2, r14,LSL #1	// initial output line address
	vmov.i64	d14,#0			// padding value
	// looped 2 more lines
	ldr		r8,[sp, #0x6c]		// r8 = line counter
	mul		r14, r8, r4		// input width * height
	sub		r8, r8, #1

	cmp		r11, #0			// if pad=1, first line is 0
	bne		more2_column_line_loop	// if pad=0, accumulate first line
	// accumulate first line
	vldr		d0, [r6]
	vldr		s2, [r6, #0x8]
        pld		[r6, r14, LSL #2]
	add		r6, r6, r4, LSL #2 
	sub		r8, r8, #1
	vmla.f32	s28, s0, s18
	vmla.f32	s28, s1, s19
	vmla.f32	s28, s2, s20

more2_column_line_loop:
	vldr		d0, [r6]
	vldr		s2, [r6, #0x8]
	vmla.f32	s28, s0, s21
        pld		[r6, r14, LSL #2]
	add		r6, r6, r4, LSL #2 
	vldr		s9, [r6]
	vmla.f32	s28, s1, s22
	vldr		d5, [r6, #0x4]
	vmla.f32	s28, s2, s23
        pld		[r6, r14, LSL #2]
	vmla.f32	s28, s9, s24
	add		r6, r6, r4, LSL #2
	vmla.f32	s28, s10,s25
	vmla.f32	s28, s11,s26
        //add bias
        vadd.f32        d14,d14,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
	vmul.f32	s28, s9, s18
	add		r7 , r7, r5, LSL #2
	vmla.f32	s28, s10,s19
	subs		r8 , r8, #2
	vmla.f32	s28, s11,s20
	bgt		more2_column_line_loop
	tst		r8, #1
	bne		more2_column_line_loop_end
	// last line
	vldm		r6, {s0 - s2}
	vmla.f32	s28, s0, s21
	vmla.f32	s28, s1, s22
	vmla.f32	s28, s2, s23
        //add bias
        vadd.f32        d14,d14,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
more2_column_line_loop_end:
	add		r9, r9, #2
more2_column_end:
    // last 1 column
        sub             r14, r4, r9
        cmp             r14, #1
	beq		last_column_end
	add		r6, r0, r9, LSL #2	// initial input line address
	add		r14,r9, r11
	add		r7, r2, r14,LSL #1	// initial output line address
	vmov.i64	d14,#0			// padding value
	// looped 2 more lines
	ldr		r8, [sp,#0x6c]		// r8 = line counter
	mul		r14, r8, r4		// input width * height
	sub		r8, r8, #1

	cmp		r11, #0			// if pad=1, first line is 0
	bne		last_column_line_loop	// if pad=0, accumulate first line
	// accumulate first line
	vldm		r6,{s0 - s1}
	pld		[r6, r14, LSL #2]
	add		r6, r6, r4, LSL #2
	sub		r8, r8, #1 
	vmla.f32	s28, s0, s18
	vmla.f32	s28, s1, s19

last_column_line_loop:
	vldm		r6,{s0 - s1}
	pld		[r6, r14, LSL #2]
	add		r6, r6, r4, LSL #2 
	vmla.f32	s28, s0, s21
	vldm		r6,{s9 - s10}
	vmla.f32	s28, s1, s22
	pld		[r6, r14, LSL #2]
	add		r6, r6, r4, LSL #2
	vmla.f32	s28, s9, s24
	vmla.f32	s28, s10,s25
        //add bias	
        vadd.f32        d14,d14,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
	vmul.f32	s28, s9, s18
	add		r7 , r7, r5, LSL #2
	vmla.f32	s28, s10,s19
	subs		r8 , r8, #2
	bgt		last_column_line_loop
	tst		r8, #1
	bne		last_column_end
	// last line
	vldm		r6,{s0 - s1}
	vmla.f32	s28, s0, s21
	vmla.f32	s28, s1, s22
        //add bias
        vadd.f32        d14,d14,d16
	#ifdef CONV_RELU_FUSE
	vmax.f32	d14, d14, d18
	#ifdef CONV_RELU6_FUSE
	vmin.f32	d14, d14, d20
	#endif
	#endif
	vstr		s28, [r7]
last_column_end:
	// set next channel input output address
	ldr		r6, [sp, #0x6c]		// input height
	mul		r7, r6, r4		// input width * height
	add		r0, r0, r7, LSL #2	// new input address
	add		r6, r6, r11
	lsr		r6, r6, #1		// output height
	mul		r7, r6, r5		// output width * height
	add		r2, r2, r7, LSL #2
	subs		r3, r3, #1
	bne		channel_loop
	// restore content
	vpop		{d8 - d15}
	pop		{r4 - r12, pc}
	.end
