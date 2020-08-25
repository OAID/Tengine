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
 * Copyright (c) 2018, OPEN AI LAB
 * Author: xiaowei@openailab.com
 */

//
// depthwise convolution kernel size 3x3 stride 2  left pad can be 0 or 1
// input:
//        x0     arg0  bias point
//        x1     arg1  input data address
//        x2     arg2  kernel data address
//        x3     arg3  output data address
//        x4     arg4  channel number
//        x5     arg5  input width     must >=4
//        x6     arg6  input height    must > 3
//        x7     arg7  left pad 1 flag
// output: no
//
// register definition
//        x0     bias_point
//        x1     intput data address for every channel
//        x2     kernel pointer
//        x3     output data address for every channel
//        x4     channel counter
//        x5     input width
//        x6     input height
//        x7     pad
//        x9     input pointer
//        x10    output pointer
//        x11    line counter
//        x12    column counter
//        x13    temp register
//        x14    next page offset
//        x15    next input line address
//        x16    output width
//
// kernel q0     k21 k20 k12 k11 k10 k02 k01 k00
//        q1     xx  xx  xx  xx  xx  xx  xx  k22
//
// input  q2 ~ q4
//        q5 ~ q7
//
// output q16 q17
//
// temp   q20 ~ q23
//        q24 ~ q27
//
// bias   q28
// relu   q29, q30

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2_fp16_a76
#endif

        .section .text, "ax"
        .align 5
        .type KERNEL_NAME STT_FUNC
        .global KERNEL_NAME
        .hidden KERNEL_NAME
KERNEL_NAME:
#ifdef CONV_RELU_FUSE
	movi	d29, 0				// zero value for relu
#ifdef CONV_RELU6_FUSE
	mov	x13, 6
	scvtf	h30, x13
	dup	v30.8h, v30.h[0]		//6 value for relu6
#endif
#endif
	movi	d28, 0
	add	x16,x5, x7
	lsr	x16,x16,1			// output_width
channel_loop:
	// load kernel
	ldr	q0, [x2]
	ldr	h1, [x2, 0x10]
	prfm	pldl2keep, [x2, 0x40]
	add	x2, x2, 0x12

	cbz	x0, no_biases	
	ld1r	{v28.8h}, [x0], 0x2
no_biases:
	mov	x9, x1				// intial channel input point
	mov	x10,x3				// intial channel output point
	mov	x12, 1				// initial input column counter

	cbz	x7, first_column_finish

	cmp	x5, 9
	ble	first4_column_start

    // first 8 column 
first8_column_start:
	mov	v16.8b, v28.8b
	sub	x11, x6,  1
	cbnz	x7, first8_column_line_loop_start//if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     q2, [x9]                        // v2 = [ i07 i06 i05 i04 i03 i02 i01 i00]
	sub	x11, x11, 1
	uzp2	v21.8h,	v2.8h,  v2.8h		// v21 = [ i07 i05 i03 i01]
        uzp1    v20.8h, v2.8h,  v2.8h           // v20 = [ i06 i04 i02 i00]
        ext	v2.8b, v29.8b, v21.8b, 6    	// v2  = [ i05 i03 i01  0 ]

	fmla	v16.4h, v20.4h, v0.h[1]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[2]
        add     x9, x9, x5, LSL 1
	fmla	v16.4h,  v2.4h, v0.h[0]		// line0 [d3  d2  d1  d0] x kernel line 0

first8_column_line_loop_start:
	add	x15,x9, x5, LSL 1

first8_column_line_loop:
        ldr     q2, [x9]                        // v2 = [ i07 i06 i05 i04 i03 i02 i01 i00]
	ldr	q5, [x15]			// v5 = [ i17 i16 i15 i14 i13 i12 i11 i10]
        subs    x11,x11, 2
	uzp2	v21.8h,	v2.8h,  v2.8h		// v21 = [ i07 i05 i03 i01]
        uzp1    v20.8h, v2.8h,  v2.8h           // v20 = [ i06 i04 i02 i00]
	uzp2	v25.8h,	v5.8h,  v5.8h		// v25 = [ i17 i15 i13 i11]
        uzp1    v24.8h, v5.8h,  v5.8h           // v24 = [ i16 i14 i12 i10]
        ext     v2.8b, v29.8b, v21.8b, 6       	// v2  = [ i05 i03 i01  0 ]
        ext     v5.8b, v29.8b, v25.8b, 6       	// v5  = [ i15 i13 i01  0 ]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[5]
        add     x9, x9, x5, LSL 2
	fmla	v16.4h, v24.4h, v0.h[7]
	fmla	v16.4h, v25.4h, v1.h[0]
	prfm	pldl1keep, [x15,0x40]
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d3  d2  d1  d0] x kernel line 1
        add     x15,x15,x5, LSL 2
	fmla	v16.4h,  v5.4h, v0.h[6]		// line0 [d3  d2  d1  d0] x kernel line 2
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]
	mov	v16.8b, v28.8b
        add     x10,x10, x16, LSL 1
        fmla    v16.4h,  v5.4h, v0.h[0]         // line0 [d1  d0] x kernel line 1
        fmla    v16.4h, v24.4h, v0.h[1]
        fmla    v16.4h, v25.4h, v0.h[2]
	bgt	first8_column_line_loop
	mov	x12,  8				// reinitial column counter

	bne	first_column_finish
// last 1 line
        ldr     q2, [x9]                        // v2 = [ i07 i06 i05 i04 i03 i02 i01 i00]
	uzp2	v21.8h,	v2.8h,  v2.8h		// v21 = [ i07 i05 i03 i01]
        uzp1    v20.8h, v2.8h,  v2.8h           // v20 = [ i06 i04 i02 i00]
        ext     v2.8b, v29.8b, v21.8b, 6       	// v2  = [ i05 i03 i01  0 ]
	fmla	v16.4h, v21.4h, v0.h[5]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v20.4h, v0.h[4]
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d3  d2  d1  d0] x kernel line 1
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]

	b	first_column_finish

    // first 4 column
first4_column_start:
	mov	v16.8b, v28.8b
	sub	x11, x6,  1
	cbnz	x7, first4_column_line_loop_start//if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	sub	x11, x11, 1
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v29.h[0]        	// v2  = [ x   x  i01  0 ]

	fmla	v16.4h, v20.4h, v0.h[1]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[2]
        add     x9, x9, x5, LSL 1
	fmla	v16.4h,  v2.4h, v0.h[0]		// line0 [d1  d0] x kernel line 0

first4_column_line_loop_start:
	add	x15,x9, x5, LSL 1

first4_column_line_loop:
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	d5, [x15]			// v5 = [ i13 i12 i11 i10]
        subs    x11,x11, 2
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        uzp1    v24.4h, v5.4h,  v5.4h           // v24 = [ x   x  i12 i10]
	uzp2	v25.4h,	v5.4h,  v5.4h		// v25 = [ x   x  i13 i11]
        ins     v2.h[0], v29.h[0]        	// v2  = [ x   x  i01  0 ]
        ins     v5.h[0], v29.h[0]        	// v5  = [ x   x  i01  0 ]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[5]
        add     x9, x9, x5, LSL 2
	fmla	v16.4h, v24.4h, v0.h[7]
	prfm	pldl1keep, [x15,0x40]
	fmla	v16.4h, v25.4h, v1.h[0]
        add     x15,x15,x5, LSL 2
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
	fmla	v16.4h,  v5.4h, v0.h[6]		// line0 [d1  d0] x kernel line 2
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
	mov	v16.8b, v28.8b
        add     x10,x10,x16, LSL 1
        fmla    v16.4h,  v5.4h, v0.h[0]         // line0 [d1  d0] x kernel line 1
        fmla    v16.4h, v24.4h, v0.h[1]
        fmla    v16.4h, v25.4h, v0.h[2]
	bgt	first4_column_line_loop
	mov	x12,  4				// reinitial column counter

	bne	first_column_finish
// last 1 line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v29.h[0]        	// v2  = [ x   x  i01  0 ]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[5]
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
first_column_finish:

    // 16 more column
more16_column_start:
	sub	x13, x5, x12
	cmp	x13, 17	// must have 17 more column to start more4 column loop
	blt	more16_column_finish
	add	x9, x1, x12, LSL 1		// initial input pointer	
	lsr	x10, x12, 1
	add	x10,x3, x10, LSL 1		// initial output pointer

	mov	v16.16b, v28.16b
	sub	x11, x6,  1
	cbnz	x7, more16_column_line_loop_start// if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     q2, [x9, -0x2]                  // v2 = [ i06 ... i0-1]
        ldr     q3, [x9,  0xe]                  // v3 = [ i0e ... i07 ]
	ldr	h4, [x9, 0x1e]			// v4 = [ xx  xx  xx  i0f]
	sub	x11, x11, 1
        uzp1    v20.8h, v2.8h,  v3.8h           // v20 = [i0d i0b i09 i07 i05 i03 i01 i0-1]
	uzp2	v21.8h,	v2.8h,  v3.8h		// v21 = [i0e i0c i0a i08 i06 i04 i02 i00]
        ext     v2.16b,v20.16b, v4.16b, 2    	// v2  = [i0f i0d i0b i09 i07 i05 i03 i01]

	fmla	v16.8h, v20.8h, v0.h[0]		// line0 [d1  d0] x kernel line 0
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.8h, v21.8h, v0.h[1]
        add     x9, x9, x5, LSL 1
	fmla	v16.8h,  v2.8h, v0.h[2]

more16_column_line_loop_start:
	add	x15,x9, x5, LSL 1

	// looped 2 more lines
more16_column_line_loop:
        ldr     q2, [x9, -0x2]                  // v2 = [ i06 ... i0-1]
        ldr     q3, [x9,  0xe]                  // v3 = [ i0e ... i07 ]
	ldr	h4, [x9, 0x1e]			// v4 = [ xx  xx  xx  i0f]
        ldr     q5, [x15,-0x2]                  // v5 = [ i16 ... i1-1]
        ldr     q6, [x15, 0xe]                  // v6 = [ i1e ... i17 ]
	ldr	h7, [x15,0x1e]			// v7 = [ xx  xx  xx  i1f]
        subs    x11,x11, 2
        uzp1    v20.8h, v2.8h,  v3.8h           // v20 = [i0d i0b i09 i07 i05 i03 i01 i0-1]
        uzp1    v24.8h, v5.8h,  v6.8h           // v24 = [i1d i1b i19 i17 i15 i13 i11 i1-1]
	uzp2	v21.8h,	v2.8h,  v3.8h		// v21 = [i0e i0c i0a i08 i06 i04 i02 i00]
	uzp2	v25.8h,	v5.8h,  v6.8h		// v25 = [i1e i1c i1a i18 i16 i14 i12 i10]
        ext     v2.16b,v20.16b, v4.16b, 2    	// v2  = [i0f i0d i0b i09 i07 i05 i03 i01]
        ext     v5.16b,v24.16b, v7.16b, 2    	// v5  = [i1f i1d i1b i19 i17 i15 i13 i11]

	fmla	v16.8h, v20.8h, v0.h[3]		// line0 [d7 d6 d5 d4 d3 d2 d1 d0] x kernel line 1
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.8h, v24.8h, v0.h[6]		// line0 [d7 d6 d5 d4 d3 d2 d1 d0] x kernel line 2
        add     x9, x9, x5, LSL 2
	fmla	v16.8h, v21.8h, v0.h[4]
	fmla	v16.8h, v25.8h, v0.h[7]
	prfm	pldl1keep, [x15,0x40]
	fmla	v16.8h,  v2.8h, v0.h[5]
        add     x15,x15,x5, LSL 2
	fmla	v16.8h,  v5.8h, v1.h[0]
#ifdef CONV_RELU_FUSE
        fmax    v16.8h, v16.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
#endif
#endif
        str     q16, [x10]
	mov	v16.16b, v28.16b
        add     x10,x10,x16, LSL 1
        fmla    v16.8h, v24.8h, v0.h[0]         // line0 [d7 d6 d5 d4 d3 d2 d1 d0] x kernel line 1
        fmla    v16.8h, v25.8h, v0.h[1]
        fmla    v16.8h,  v5.8h, v0.h[2]
	bgt	more16_column_line_loop
	add	x12, x12, 16			// update column counter

	bne	more16_column_start
        ldr     q2, [x9, -0x2]                  // v2 = [ i06 ... i0-1]
        ldr     q3, [x9,  0xe]                  // v3 = [ i0e ... i07 ]
	ldr	h4, [x9, 0x1e]			// v4 = [ xx  xx  xx  i0f]
        uzp1    v20.8h, v2.8h,  v3.8h           // v20 = [i0d i0b i09 i07 i05 i03 i01 i0-1]
	uzp2	v21.8h,	v2.8h,  v3.8h		// v21 = [i0e i0c i0a i08 i06 i04 i02 i00]
        ext     v2.16b,v20.16b, v4.16b, 2    	// v2  = [i0f i0d i0b i09 i07 i05 i03 i01]

	fmla	v16.8h, v20.8h, v0.h[3]		// line0 [d7 d6 d5 d4 d3 d2 d1 d0] x kernel line 1
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.8h, v21.8h, v0.h[4]
	fmla	v16.8h,  v2.8h, v0.h[5]
#ifdef CONV_RELU_FUSE
        fmax    v16.8h, v16.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
#endif
#endif
        str     q16, [x10]
	b	more16_column_start

more16_column_finish:

    // 4 more column
more4_column_start:
	sub	x13, x5, x12
	cmp	x13, 5	// must have 5 more column to start more4 column loop
	blt	more4_column_finish
	add	x9, x1, x12, LSL 1		// initial input pointer	
	lsr	x10, x12, 1
	add	x10,x3, x10, LSL 1		// initial output pointer

	mov	v16.8b, v28.8b
	sub	x11, x6,  1
	cbnz	x7, more4_column_line_loop_start// if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
	sub	x11, x11, 1
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]

	fmla	v16.4h, v20.4h, v0.h[1]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[2]
        add     x9, x9, x5, LSL 1
	fmla	v16.4h,  v2.4h, v0.h[0]		// line0 [d1  d0] x kernel line 0

more4_column_line_loop_start:
	add	x15,x9, x5, LSL 1

	// looped 2 more lines
more4_column_line_loop:
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
	ldr	d5, [x15]			// v5 = [ i13 i12 i11 i10]
	ldr	h6, [x15,-0x2]			// v6 = [ xx  xx  xx  i1-1]
        subs    x11,x11, 2
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]
        uzp1    v24.4h, v5.4h,  v5.4h           // v24 = [ x   x  i12 i10]
	uzp2	v25.4h,	v5.4h,  v5.4h		// v25 = [ x   x  i13 i11]
        ins     v5.h[0], v6.h[0]        	// v5  = [ x   x  i01 i1-1]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[5]
        add     x9, x9, x5, LSL 2
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
	fmla	v16.4h, v24.4h, v0.h[7]
	prfm	pldl1keep, [x15,0x40]
	fmla	v16.4h, v25.4h, v1.h[0]
        add     x15,x15,x5, LSL 2
	fmla	v16.4h,  v5.4h, v0.h[6]		// line0 [d1  d0] x kernel line 2
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
	mov	v16.8b, v28.8b
        add     x10,x10,x16, LSL 1
        fmla    v16.4h,  v5.4h, v0.h[0]         // line0 [d1  d0] x kernel line 1
        fmla    v16.4h, v24.4h, v0.h[1]
        fmla    v16.4h, v25.4h, v0.h[2]
	bgt	more4_column_line_loop
	add	x12, x12, 4			// update column counter

	bne	more4_column_start
// last 1 line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h, v21.4h, v0.h[5]
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]

	b	more4_column_start

more4_column_finish:
	mul	x14, x5, x6			// x14 is used as next page offset
	sub	x14, x14,x5
	lsl	x14, x14, 1
	add	x14, x14, 10
	sub	x13, x5, x12

	cmp	x13, 4
	beq	last4_column_start
	cmp	x13, 3
	beq	last3_column_start		// last 3 column
	cmp	x13, 2
	beq	last2_column_start		// last 2 column
	b	last1_column_start		// last 1 column

last2_column_start:
	sub	x12, x12, 2
	sub	x14, x14, 4
last4_column_start:
	add	x9, x1, x12, LSL 1	
	lsr	x10, x12, 1
	add	x10,x3, x10, LSL 1		// initial output pointer

	mov	v16.8b, v28.8b
	sub	x11, x6,  1
	cbnz	x7, last4_column_line_loop_start// if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
	sub	x11, x11, 1
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]

	fmla	v16.4h, v20.4h, v0.h[1]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v21.4h, v0.h[2]
        add     x9, x9, x5, LSL 1
	fmla	v16.4h,  v2.4h, v0.h[0]		// line0 [d1  d0] x kernel line 0

last4_column_line_loop_start:
	add	x15,x9, x5, LSL 1

	// looped 2 more lines
last4_column_line_loop:
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
	ldr	d5, [x15]			// v5 = [ i13 i12 i11 i10]
	ldr	h6, [x15,-0x2]			// v6 = [ xx  xx  xx  i1-1]
        subs    x11,x11, 2
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]
        uzp1    v24.4h, v5.4h,  v5.4h           // v24 = [ x   x  i12 i10]
	uzp2	v25.4h,	v5.4h,  v5.4h		// v25 = [ x   x  i13 i11]
        ins     v5.h[0], v6.h[0]        	// v5  = [ x   x  i01 i1-1]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v21.4h, v0.h[5]
        add     x9, x9, x5, LSL 2
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
	fmla	v16.4h, v24.4h, v0.h[7]
	prfm	pldl1keep, [x15,x14]
	fmla	v16.4h, v25.4h, v1.h[0]
        add     x15,x15,x5, LSL 2
	fmla	v16.4h,  v5.4h, v0.h[6]		// line0 [d1  d0] x kernel line 2
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
	mov	v16.8b, v28.8b
        add     x10,x10,x16, LSL 1
        fmla    v16.4h,  v5.4h, v0.h[0]         // line0 [d1  d0] x kernel line 1
        fmla    v16.4h, v24.4h, v0.h[1]
        fmla    v16.4h, v25.4h, v0.h[2]
	bgt	last4_column_line_loop

	bne	channel_end
// last 1 line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
	ldr	h3, [x9, -0x2]			// v3 = [ xx  xx  xx  i0-1]
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i02 i00]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i03 i01]
        ins     v2.h[0], v3.h[0]        	// v2  = [ x   x  i01 i0-1]
	fmla	v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v21.4h, v0.h[5]
	fmla	v16.4h,  v2.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
	b	channel_end

last1_column_start:
	sub	x12, x12, 2
	sub	x14, x14, 4
last3_column_start:
	add	x9, x1, x12, LSL 1	
	lsr	x10, x12, 1
	add	x10,x3, x10, LSL 1		// initial output pointer

	mov	v16.8b, v28.8b
	sub	x11, x6,  1
	cbnz	x7, last3_column_line_loop_start// if pad=1, first line is 0
						// if pad=0, accumulate first line
        ldr     d2, [x9, -0x2]                  // v2 = [ i02 i01 i00 i0-1]
	ldr	h3, [x9, 0x2]			// v3 = [ xx  xx   0  i01]
	sub	x11, x11, 1
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i01 i0-1]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i02 i00]

	fmla	v16.4h,  v3.4h, v0.h[2]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v20.4h, v0.h[0]		// line0 [d1  d0] x kernel line 0
        add     x9, x9, x5, LSL 1
	fmla	v16.4h, v21.4h, v0.h[1]

last3_column_line_loop_start:
	add	x15,x9, x5, LSL 1

	// looped 2 more lines
last3_column_line_loop:
        ldr     d2, [x9, -0x2]                  // v2 = [ i02 i01 i00 i0-1]
	ldr	h3, [x9, 0x2]			// v3 = [ xx  xx   0  i01]
	ldr	d5, [x15,-0x2]			// v5 = [ i12 i11 i10 i1-1]
	ldr	h6, [x15,0x2]			// v6 = [ xx  xx   0  i11]
        subs    x11,x11, 2
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i01 i0-1]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i02 i00]
        uzp1    v24.4h, v5.4h,  v5.4h           // v24 = [ x   x  i11 i1-1]
	uzp2	v25.4h,	v5.4h,  v5.4h		// v25 = [ x   x  i12 i10]
	fmla	v16.4h,  v3.4h, v0.h[5]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v20.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
        add     x9, x9, x5, LSL 2
	fmla	v16.4h, v21.4h, v0.h[4]
	fmla	v16.4h,  v6.4h, v1.h[0]
	prfm	pldl1keep, [x15,x14]
	fmla	v16.4h, v24.4h, v0.h[6]		// line0 [d1  d0] x kernel line 2
        add     x15,x15,x5, LSL 2
	fmla	v16.4h, v25.4h, v0.h[7]
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]
	mov	v16.8b, v28.8b
        add     x10,x10, x16, LSL 1
        fmla    v16.4h, v24.4h, v0.h[0]         // line0 [d1  d0] x kernel line 1
        fmla    v16.4h, v25.4h, v0.h[1]
        fmla    v16.4h,  v6.4h, v0.h[2]
	bgt	last3_column_line_loop
	bne	channel_end
// last 1 line
        ldr     d2, [x9, -0x2]                  // v2 = [ i02 i01 i00 i0-1]
	ldr	h3, [x9, 0x2]			// v3 = [ xx  xx   0  i01]
        uzp1    v20.4h, v2.4h,  v2.4h           // v20 = [ x   x  i01 i0-1]
	uzp2	v21.4h,	v2.4h,  v2.4h		// v21 = [ x   x  i02 i00]
	fmla	v16.4h,  v3.4h, v0.h[5]
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h, v20.4h, v0.h[3]		// line0 [d1  d0] x kernel line 1
	fmla	v16.4h, v21.4h, v0.h[4]
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     s16, [x10]

channel_end:
	subs	x4, x4, 1
	mul	x13,x5, x6
	add	x1, x1, x13, LSL 1		// next channel input

	add	x14,x6, x7
	lsr	x14,x14,1	
	mul	x13,x16,x14
	add	x3, x3, x13, LSL 1
	bne	channel_loop

	ret
