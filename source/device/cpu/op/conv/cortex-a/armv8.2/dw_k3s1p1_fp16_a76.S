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
// depthwise convolution kernel size 3x3 stride 1  pad 1
// input:
//        x0     arg0  input data address
//        x1     arg1  kernel data address
//        x2     arg2  output data address
//        x3     arg3  channel number
//        x4     arg4  input width     must >=4
//        x5     arg5  input height
//        x6     arg6  bias point
// output: no
//
// register definition
//        x0     intput data address for every channel
//        x1     kernel pointer
//        x2     output data address for every channel
//        x3     channel counter
//        x4     input width
//        x5     input height
//        x6     bias_point
//        x9     input pointer
//        x10    output pointer
//        x11    line counter
//        x12    column counter
//        x13    temp register
//        x14    next page offset
//        x15    next input line address
//
// kernel q0     k21 k20 k12 k11 k10 k02 k01 k00
//        q1     xx  xx  xx  xx  xx  xx  xx  k22
//
// input  q2 ~ q4
//        q5 ~ q7
//
// output line0 q16 q17
//        line1 q18 q19
//
// temp   q20 ~ q23
//        q24 ~ q27
//
// bias   q28
// relu   q29, q30

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1_fp16_a76
#endif

        .section .text, "ax"
        .align 5
        .type KERNEL_NAME STT_FUNC
        .global KERNEL_NAME
        .hidden KERNEL_NAME
KERNEL_NAME:
#ifdef CONV_RELU_FUSE
	movi	d29, 0
#ifdef CONV_RELU6_FUSE
	mov	x13, 6
	scvtf	h30, x13
	dup	v30.8h, v30.h[0]
#endif
#endif
channel_loop:
	movi	d28, 0
	cbz	x6, no_biases	
	ld1r	{v28.8h}, [x6], 0x2
no_biases:
	mov	x9, x0				// intial channel input point
	mov	x10,x2				// intial channel output point
	mov	x12, 0				// initial column counter
	// load kernel
	ldr	q0, [x1]
	ldr	h1, [x1, 0x10]
	prfm	pldl2keep, [x1, 0x40]
	add	x1, x1, 0x12

        cmp     x4, 9
        blt     first4_column_start             // if input width >= 9 use long start

first8_column_start:
    // first line
        ldr     q2, [x9]                        // v2 = [i07 i06 i05 i04 i03 i02 i01 i00]
        ldr     h3, [x9, 0x10]                  // v3 = [ 0   0   0   0   0   0   0  i08]
        sub     x11, x5, 1
        ext     v20.16b,v3.16b, v2.16b, 14      // v20 = [i06 i05 i04 i03 i02 i01 i00  0 ]
        cmp     x11, 3
        ext     v21.16b,v2.16b, v3.16b, 2       // v21 = [i08 i07 i06 i05 i04 i03 i02 i01]
        fmul    v16.8h, v20.8h, v0.h[3]         // line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line1
        fmul    v18.8h, v20.8h, v0.h[0]         // line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line0
	prfm	pldl1keep, [x9, 0x40]
        fmla    v16.8h,  v2.8h, v0.h[4]
        add     x9, x9, x4, LSL 1
        fmla    v18.8h,  v2.8h, v0.h[1]
        fmla    v16.8h, v21.8h, v0.h[5]
        fmla    v18.8h, v21.8h, v0.h[2]
        blt     first8_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 1

        // looped 2 more lines
first8_column_line_loop:
        // line 1
        ldr     q5, [x9]                        // v5 = [i17 i16 i15 i14 i13 i12 i11 i10]
        ldr     h6, [x9, 0x10]                  // v6 = [ 0   0   0    0   0  0   0  i18]
        ldr     q2, [x15]                       // v2 = [i27 i26 i25 i24 i23 i22 i21 i20]
        ldr     h3, [x15,0x10]                  // v3 = [ 0   0   0   0   0   0   0  i28]
        subs    x11,x11, 1
        ext     v20.16b,v6.16b, v5.16b, 14      // v20 = [i16 i15 i14 i13 i12 i11 i10  0 ]
        ext     v21.16b,v5.16b, v6.16b, 2       // v21 = [i18 i17 i16 i15 i14 i13 i12 i11]
        ext     v24.16b,v3.16b, v2.16b, 14      // v24 = [i26 i25 i24 i23 i22 i21 i20  0 ]
        ext     v25.16b,v2.16b, v3.16b, 2       // v25 = [i28 i27 i26 i25 i24 i23 i22 i21]

	prfm	pldl1keep, [x9, 0x40]
        fmla    v16.8h, v20.8h, v0.h[6]         // old line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
        fmla    v18.8h, v20.8h, v0.h[3]         // old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
	prfm	pldl1keep, [x15,0x40]
        fmla    v16.8h,  v5.8h, v0.h[7]
        fmla    v18.8h,  v5.8h, v0.h[4]
        fmla    v16.8h, v21.8h, v1.h[0]
        fmla    v18.8h, v21.8h, v0.h[5]

        fadd    v16.8h, v16.8h, v28.8h
#ifdef CONV_RELU_FUSE
        fmax    v16.8h, v16.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
#endif
#endif
        str     q16, [x10]
        add     x10,x10, x4, LSL 1

        fmla    v18.8h, v24.8h, v0.h[6]         // old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
        fmul    v16.8h, v20.8h, v0.h[0]         // new line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 0
        add     x9, x9, x4, LSL 2
        fmla    v18.8h, v2.8h,  v0.h[7]
        fmla    v16.8h, v5.8h,  v0.h[1]
        add     x15,x15,x4, LSL 2
        fmla    v18.8h, v25.8h, v1.h[0]
        fmla    v16.8h, v21.8h, v0.h[2]

        fadd    v18.8h, v18.8h, v28.8h
#ifdef CONV_RELU_FUSE
        fmax    v18.8h, v18.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v18.8h, v18.8h, v30.8h
#endif
#endif
        str     q18, [x10]
        add     x10,x10,x4, LSL 1

        fmla    v16.8h, v24.8h, v0.h[3]         // new line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
        fmul    v18.8h, v24.8h, v0.h[0]         // new line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 0
        fmla    v16.8h, v2.8h,  v0.h[4]
        fmla    v18.8h, v2.8h,  v0.h[1]
        fmla    v16.8h, v25.8h, v0.h[5]
        fmla    v18.8h, v25.8h, v0.h[2]

        bne     first8_column_line_loop
first8_column_line_loop_end:
        and     x11, x5, 0x1
        add     x12, x12, 8
        cbnz    x11, first8_column_line_last0

        ldr     q5, [x9]                        // v5 = [i17 i16 i15 i14 i13 i12 i11 i10]
        ldr     h6, [x9, 0x10]                  // v6 = [ 0   0   0    0   0  0   0  i18]
        ext     v20.16b,v6.16b, v5.16b, 14      // v20 = [i16 i15 i14 i13 i12 i11 i10  0 ]
        ext     v21.16b,v5.16b, v6.16b, 2       // v21 = [i18 i17 i16 i15 i14 i13 i12 i11]
	prfm	pldl1keep, [x9, 0x40]
        fmla    v16.8h, v20.8h, v0.h[6]         // old line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
        fmla    v18.8h, v20.8h, v0.h[3]         // old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
        fmla    v16.8h, v5.8h,  v0.h[7]
        fmla    v18.8h, v5.8h,  v0.h[4]
        fmla    v16.8h, v21.8h, v1.h[0]
        fmla    v18.8h, v21.8h, v0.h[5]

        fadd    v16.8h, v16.8h, v28.8h
        fadd    v18.8h, v18.8h, v28.8h
#ifdef CONV_RELU_FUSE
        fmax    v16.8h, v16.8h, v29.8h
        fmax    v18.8h, v18.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
	fmin	v18.8h, v18.8h, v30.8h
#endif
#endif
        str     q16, [x10]
        add     x10,x10, x4, LSL 1
        str     q18, [x10]
        b       first_column_finish
first8_column_line_last0:
        fadd    v16.8h, v16.8h, v28.8h
#ifdef CONV_RELU_FUSE
        fmax    v16.8h, v16.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
#endif
#endif
        str     q16, [x10]
        b       first_column_finish

    // first 4 column
first4_column_start:
    // first line
        ldr     d2, [x9]                        // v2 = [ i03 i02 i01 i00]
        ldr     h3, [x9, 0x8]                   // v3 = [  0   0   0  i04]
        sub     x11, x5, 1
        ext     v20.8b,v3.8b, v2.8b, 6          // v20 = [i02 i01 i00  0 ]
        cmp     x11, 3
        ext     v21.8b,v2.8b, v3.8b, 2          // v21 = [i04 i03 i02 i01]
	prfm	pldl1keep, [x9, 0x40]
        fmul    v16.4h, v20.4h, v0.h[3]         // line0 [d3  d2  d1  d0] x kernel line 1
        fmul    v18.4h, v20.4h, v0.h[0]         // line1 [d3  d2  d1  d0] x kernel line 0
        add     x9, x9, x4, LSL 1
        fmla    v16.4h,  v2.4h, v0.h[4]
        fmla    v18.4h,  v2.4h, v0.h[1]
        fmla    v16.4h, v21.4h, v0.h[5]
        fmla    v18.4h, v21.4h, v0.h[2]
	blt	first4_column_line_loop_end
	lsr	x11, x11, 1
	add	x15,x9, x4, LSL 1 

	// looped 2 more lines
first4_column_line_loop:
	// line 1
        ldr     d5, [x9]                        // v5 = [i13 i12 i11 i10]
        ldr     h6, [x9, 0x8]                   // v6 = [  0  0   0  i14]
        ldr     d2, [x15]                       // v2 = [i23 i22 i21 i20]
        ldr     h3, [x15,0x8]                   // v3 = [ 0   0   0  i24]
        subs    x11,x11, 1
        ext     v20.8b,v6.8b, v5.8b, 6          // v20= [i12 i11 i10  0 ]
        ext     v21.8b,v5.8b, v6.8b, 2          // v21= [i14 i13 i12 i11]
        ext     v24.8b,v3.8b, v2.8b, 6          // v24= [i22 i21 i20  0 ]
        ext     v25.8b,v2.8b, v3.8b, 2          // v25= [i24 i23 i22 i21]

        fmla    v16.4h, v20.4h, v0.h[6]         // old line0 [d3  d2  d1  d0] x kernel line 2
        fmla    v18.4h, v20.4h, v0.h[3]         // old line1 [d3  d2  d1  d0] x kernel line 1
        fmla    v16.4h,  v5.4h, v0.h[7]
	prfm	pldl1keep, [x9, 0x40]
        fmla    v18.4h,  v5.4h, v0.h[4]
        fmla    v16.4h, v21.4h, v1.h[0]
	prfm	pldl1keep, [x15,0x40]
        fmla    v18.4h, v21.4h, v0.h[5]

        fadd    v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1

        fmla    v18.4h, v24.4h, v0.h[6]         // old line1 [d3  d2  d1  d0] x kernel line 2
        fmul    v16.4h, v20.4h, v0.h[0]         // new line0 [d3  d2  d1  d0] x kernel line 0
        fmla    v18.4h,  v2.4h, v0.h[7]
        add     x9, x9, x4, LSL 2
        fmla    v16.4h,  v5.4h, v0.h[1]
        add     x15,x15,x4, LSL 2
        fmla    v18.4h, v25.4h, v1.h[0]
        fmla    v16.4h, v21.4h, v0.h[2]

        fadd    v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
        fmax    v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d18, [x10]
        add     x10,x10,x4, LSL 1

        fmla    v16.4h, v24.4h, v0.h[3]         // new line0 [d3  d2  d1  d0] x kernel line 1
        fmul    v18.4h, v24.4h, v0.h[0]         // new line1 [d3  d2  d1  d0] x kernel line 0
        fmla    v16.4h,  v2.4h, v0.h[4]
        fmla    v18.4h,  v2.4h, v0.h[1]
        fmla    v16.4h, v25.4h, v0.h[5]
        fmla    v18.4h, v25.4h, v0.h[2]

	bne	first4_column_line_loop

first4_column_line_loop_end:
	and	x11, x5, 0x1
	add	x12, x12, 4			// update column counter
	cbnz	x11, first4_column_line_last0
	// last 1 line
        ldr     d5, [x9]                        // v5 = [ii13 i12 i11 i10]
        ldr     h6, [x9, 0x8]                   // v6 = [   0  0   0  i14]
        ext     v20.8b,v6.8b, v5.8b, 6          // v20 = [i12 i11 i10  0 ]
        ext     v21.8b,v5.8b, v6.8b, 2          // v21 = [i14 i13 i12 i11]
	prfm	pldl1keep, [x9, 0x40]
        fmla    v16.4h, v20.4h, v0.h[6]         // old line0 [d3  d2  d1  d0] x kernel line 2
        fmla    v18.4h, v20.4h, v0.h[3]         // old line1 [d3  d2  d1  d0] x kernel line 1
        fmla    v16.4h, v5.4h,  v0.h[7]
        fmla    v18.4h, v5.4h,  v0.h[4]
        fmla    v16.4h, v21.4h, v1.h[0]
        fmla    v18.4h, v21.4h, v0.h[5]

        fadd    v16.4h, v16.4h, v28.4h
        fadd    v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
        fmax    v16.4h, v16.4h, v29.4h
        fmax    v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1
        str     d18, [x10]
	b	first_column_finish

first4_column_line_last0:
        fadd    v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax    v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]

first_column_finish:
	
    // 16 more column
more16_column_start:
	sub	x13, x4, x12
	cmp	x13, 17	// must have 17 more column to start more16 column loop
	blt	more16_column_finish
	add	x9, x0, x12, LSL 1
	add	x10,x2, x12, LSL 1		// initial output pointer
	sub	x9, x9, 2			// initial input pointer
	// first line
        ldp     q2, q3, [x9]        // v3 = [i0e i0d i0c i0b i0a i09 i08 i07] v2 = [i06 i05 i04 i03 i02 i01 i00 i0-1]
        ldr     s4, [x9, 0x20]      // v4 = [ 0   0  i0g i0f ]
	sub	x11, x5, 1
        ext     v20.16b,v2.16b, v3.16b, 2       // v20 = [i07 i06 i05 i04 i03 i02 i01 i00]
	cmp	x11, 3
        ext     v21.16b,v2.16b, v3.16b, 4       // v21 = [i08 i07 i06 i05 i04 i03 i02 i01]
        ext     v24.16b,v3.16b, v4.16b, 2       // v24 = [i0f i0e i0d i0c i0b i0a i09 i08]
        ext     v25.16b,v3.16b, v4.16b, 4       // v25 = [i0g i0f i0e i0d i0c i0b i0a i09]
        fmul    v16.8h, v2.8h,  v0.h[3]         // line0 [ d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
        fmul    v17.8h, v3.8h,  v0.h[3]         // line0 [ df  de  dd  dc  db  da  a9  d8] x kernel line 1
        fmul    v18.8h, v2.8h,  v0.h[0]         // line1 [ d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 0
        fmul    v19.8h, v3.8h,  v0.h[0]         // line1 [ df  de  dd  dc  db  da  a9  d8] x kernel line 0
        fmla    v16.8h, v20.8h, v0.h[4]
	prfm	pldl1keep, [x9, 0x40]
        fmla    v17.8h, v24.8h, v0.h[4]
        fmla    v18.8h, v20.8h, v0.h[1]
        fmla    v19.8h, v24.8h, v0.h[1]
        add     x9, x9, x4, LSL 1
        fmla    v16.8h, v21.8h, v0.h[5]
        fmla    v17.8h, v25.8h, v0.h[5]
        fmla    v18.8h, v21.8h, v0.h[2]
        fmla    v19.8h, v25.8h, v0.h[2]
        blt     more16_column_line_loop_end
        lsr     x11, x11, 1
        add     x15, x9, x4, LSL 1
	// looped 2 more lines
more16_column_line_loop:
        // line 1
        ldp     q5, q6, [x9]       // v6 = [i1e i1d i1c i1b i1a i19 i18 i17]  v5 = [i16 i15 i14 i13 i12 i11 i10 i1-1]
        ldr     s7, [x9, 0x20]     // v7 = [ 0   0  i1g i1f ]
        ldp     q2, q3, [x15]      // v3 = [i2e i2d i2c i2b i2a i29 i28 i27]  v2 = [i26 i25 i24 i23 i22 i21 i20 i2-1]
        ldr     s4, [x15,0x20]     // v4 = [ 0   0  i2g i2f ]
        subs    x11,x11, 1
        ext     v20.16b,v5.16b, v6.16b, 2       // v20 = [i17 i16 i15 i14 i13 i12 i11 i10]
        ext     v21.16b,v5.16b, v6.16b, 4       // v21 = [i18 i17 i16 i15 i14 i13 i12 i11]
        ext     v22.16b,v6.16b, v7.16b, 2       // v22 = [i1f i1e i1d i1c i1b i1a i19 i18]
        ext     v23.16b,v6.16b, v7.16b, 4       // v23 = [i1g i1f i1e i1d i1c i1b i1a i19]
        ext     v24.16b,v2.16b, v3.16b, 2       // v24 = [i27 i26 i25 i24 i23 i22 i21 i20]
        ext     v25.16b,v2.16b, v3.16b, 4       // v25 = [i28 i27 i26 i25 i24 i23 i22 i21]
        ext     v26.16b,v3.16b, v4.16b, 2       // v26 = [i2f i2e i2d i2c i2b i2a i29 i28]
        ext     v27.16b,v3.16b, v4.16b, 4       // v27 = [i2g i2f i2e i2d i2c i2b i2a i29]
	fmla	v16.8h, v5.8h,  v0.h[6]		// old line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
	fmla	v17.8h, v6.8h,  v0.h[6]		// old line0 [df  de  dd  dc  db  da  d9  d8] x kernel line 2
	fmla	v18.8h, v5.8h,  v0.h[3]		// old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
	fmla	v19.8h, v6.8h,  v0.h[3]		// old line1 [df  de  dd  dc  db  da  d9  d8] x kernel line 1
	fmla	v16.8h,v20.8h,  v0.h[7]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v17.8h,v22.8h,  v0.h[7]
	fmla	v18.8h,v20.8h,  v0.h[4]
	fmla	v19.8h,v22.8h,  v0.h[4]
	prfm	pldl1keep, [x15,0x40]
	fmla	v16.8h,v21.8h,  v1.h[0]
	fmla	v17.8h,v23.8h,  v1.h[0]
	fmla	v18.8h,v21.8h,  v0.h[5]
	fmla	v19.8h,v23.8h,  v0.h[5]

	fadd	v16.8h, v16.8h, v28.8h
	fadd	v17.8h, v17.8h, v28.8h
#ifdef CONV_RELU_FUSE
	fmax	v16.8h, v16.8h, v29.8h
	fmax	v17.8h, v17.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
	fmin	v17.8h, v17.8h, v30.8h
#endif
#endif
        stp     q16, q17, [x10]
        add     x10,x10, x4, LSL 1

	fmul	v16.8h, v5.8h,  v0.h[0]		// new line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 0
	fmul	v17.8h, v6.8h,  v0.h[0]		// new line0 [df  de  dd  dc  db  da  d9  d8] x kernel line 0
	fmla	v18.8h, v2.8h,  v0.h[6]		// old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
	fmla	v19.8h, v3.8h,  v0.h[6]		// old line1 [df  de  dd  dc  db  da  d9  d8] x kernel line 2
	fmla	v16.8h,v20.8h,  v0.h[1]
        add     x9,  x9,  x4, LSL 2
	fmla	v17.8h,v22.8h,  v0.h[1]
        add     x15, x15, x4, LSL 2
	fmla	v18.8h,v24.8h,  v0.h[7]
	fmla	v19.8h,v26.8h,  v0.h[7]
	fmla	v16.8h,v21.8h,  v0.h[2]
	fmla	v17.8h,v23.8h,  v0.h[2]
	fmla	v18.8h,v25.8h,  v1.h[0]
	fmla	v19.8h,v27.8h,  v1.h[0]

	fadd	v18.8h, v18.8h, v28.8h
	fadd	v19.8h, v19.8h, v28.8h
#ifdef CONV_RELU_FUSE
	fmax	v18.8h, v18.8h, v29.8h
	fmax	v19.8h, v19.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v18.8h, v18.8h, v30.8h
	fmin	v19.8h, v19.8h, v30.8h
#endif
#endif
        stp     q18, q19, [x10]
        add     x10,x10,x4, LSL 1

	fmla	v16.8h, v2.8h,  v0.h[3]		// new line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
	fmla	v17.8h, v3.8h,  v0.h[3]		// new line0 [df  de  dd  dc  db  da  d9  d8] x kernel line 1
	fmul	v18.8h, v2.8h,  v0.h[0]		// new line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 0
	fmul	v19.8h, v3.8h,  v0.h[0]		// new line1 [df  de  dd  dc  db  da  d9  d8] x kernel line 0
	fmla	v16.8h,v24.8h,  v0.h[4]
	fmla	v17.8h,v26.8h,  v0.h[4]
	fmla	v18.8h,v24.8h,  v0.h[1]
	fmla	v19.8h,v26.8h,  v0.h[1]
	fmla	v16.8h,v25.8h,  v0.h[5]
	fmla	v17.8h,v27.8h,  v0.h[5]
	fmla	v18.8h,v25.8h,  v0.h[2]
	fmla	v19.8h,v27.8h,  v0.h[2]
        bne     more16_column_line_loop

more16_column_line_loop_end:
	add	x12, x12, 16			// update column counter
	and	x11, x5, 0x1
	cbnz	x11, more16_column_line_last0

        // line 1
        ldp     q5, q6, [x9]       // v6 = [i1e i1d i1c i1b i1a i19 i18 i17]  v5 = [i16 i15 i14 i13 i12 i11 i10 i1-1]
        ldr     s7, [x9, 0x20]     // v7 = [ 0   0  i1g i1f ]
        ext     v20.16b,v5.16b, v6.16b, 2       // v20 = [i17 i16 i15 i14 i13 i12 i11 i10]
        ext     v21.16b,v5.16b, v6.16b, 4       // v21 = [i18 i17 i16 i15 i14 i13 i12 i11]
        ext     v22.16b,v6.16b, v7.16b, 2       // v22 = [i1f i1e i1d i1c i1b i1a i19 i18]
        ext     v23.16b,v6.16b, v7.16b, 4       // v23 = [i1g i1f i1e i1d i1c i1b i1a i19]
	fmla	v16.8h, v5.8h,  v0.h[6]		// old line0 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 2
	fmla	v17.8h, v6.8h,  v0.h[6]		// old line0 [df  de  dd  dc  db  da  d9  d8] x kernel line 2
	fmla	v18.8h, v5.8h,  v0.h[3]		// old line1 [d7  d6  d5  d4  d3  d2  d1  d0] x kernel line 1
	fmla	v19.8h, v6.8h,  v0.h[3]		// old line1 [df  de  dd  dc  db  da  d9  d8] x kernel line 1
	fmla	v16.8h,v20.8h,  v0.h[7]
	fmla	v17.8h,v22.8h,  v0.h[7]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v18.8h,v20.8h,  v0.h[4]
	fmla	v19.8h,v22.8h,  v0.h[4]
	fmla	v16.8h,v21.8h,  v1.h[0]
	fmla	v17.8h,v23.8h,  v1.h[0]
	fmla	v18.8h,v21.8h,  v0.h[5]
	fmla	v19.8h,v23.8h,  v0.h[5]

	fadd	v16.8h, v16.8h, v28.8h
	fadd	v17.8h, v17.8h, v28.8h
	fadd	v18.8h, v18.8h, v28.8h
	fadd	v19.8h, v19.8h, v28.8h
#ifdef CONV_RELU_FUSE
	fmax	v16.8h, v16.8h, v29.8h
	fmax	v17.8h, v17.8h, v29.8h
	fmax	v18.8h, v18.8h, v29.8h
	fmax	v19.8h, v19.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
	fmin	v17.8h, v17.8h, v30.8h
	fmin	v18.8h, v18.8h, v30.8h
	fmin	v19.8h, v19.8h, v30.8h
#endif
#endif
        stp     q16, q17, [x10]
        add     x10,x10, x4, LSL 1
        stp     q18, q19, [x10]

        b       more16_column_start

more16_column_line_last0:
	fadd	v16.8h, v16.8h, v28.8h
	fadd	v17.8h, v17.8h, v28.8h
#ifdef CONV_RELU_FUSE
	fmax	v16.8h, v16.8h, v29.8h
	fmax	v17.8h, v17.8h, v29.8h
#ifdef CONV_RELU6_FUSE
	fmin	v16.8h, v16.8h, v30.8h
	fmin	v17.8h, v17.8h, v30.8h
#endif
#endif
        stp     q16, q17, [x10]
	b	more16_column_start

more16_column_finish:

    // 4 more column
more4_column_start:
	sub	x13, x4, x12
	cmp	x13, 5	// must have 5 more column to start more4 column loop
	blt	more4_column_finish
	add	x9, x0, x12, LSL 1	
	add	x10,x2, x12, LSL 1		// initial output pointer
	sub	x9, x9, 2			// initial input pointer
	// first line
        ldr     d2, [x9]            		// v2 = [i02 i01 i00 i0-1]
        ldr     s3, [x9, 0x8]      		// v3 = [ 0   0  i04 i03 ]
        sub     x11, x5, 1
        ext     v20.8b, v2.8b, v3.8b, 2         // v20 = [i03 i02 i01 i00]
        cmp     x11, 3
        ext     v21.8b, v2.8b, v3.8b, 4         // v21 = [i04 i03 i02 i01]
        fmul    v16.4h,  v2.4h, v0.h[3]         // line0 [d3  d2  d1  d0] x kernel line 1
        fmul    v18.4h,  v2.4h, v0.h[0]         // line1 [d3  d2  d1  d0] x kernel line 0
	prfm	pldl1keep, [x9, 0x40]
        fmla    v16.4h, v20.4h, v0.h[4]
        fmla    v18.4h, v20.4h, v0.h[1]
        add     x9, x9, x4, LSL 1
        fmla    v16.4h, v21.4h, v0.h[5]
        fmla    v18.4h, v21.4h, v0.h[2]
        blt     more4_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 1
	
	// looped 2 more lines
more4_column_line_loop:
        // line 1
        ldr     d5, [x9]   	    		// v5 = [i12 i11 i10 i1-1]
        ldr     s6, [x9, 0x8]     		// v6 = [ 0   0  i14 i13 ]
        ldr     d2, [x15]      			// v2 = [i22 i21 i20 i2-1]
        ldr     s3, [x15,0x8]     		// v3 = [ 0   0  i24 i23 ]
        subs    x11,x11, 1
        ext     v20.8b, v5.8b,  v6.8b, 2        // v20 = [i13 i12 i11 i10]
        ext     v21.8b, v5.8b,  v6.8b, 4        // v21 = [i14 i13 i12 i11]
        ext     v24.8b, v2.8b,  v3.8b, 2        // v24 = [i23 i22 i21 i20]
        ext     v25.8b, v2.8b,  v3.8b, 4        // v25 = [i24 i23 i22 i21]
	fmla	v16.4h, v5.4h,  v0.h[6]		// old line0 [d3  d2  d1  d0] x kernel line 2
	fmla	v18.4h, v5.4h,  v0.h[3]		// old line1 [d3  d2  d1  d0] x kernel line 1
	prfm	pldl1keep, [x9, 0x40]
	fmla	v16.4h,v20.4h,  v0.h[7]
	fmla	v18.4h,v20.4h,  v0.h[4]
	fmla	v16.4h,v21.4h,  v1.h[0]
	prfm	pldl1keep, [x15,0x40]
	fmla	v18.4h,v21.4h,  v0.h[5]

	fadd	v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1

	fmla	v18.4h, v2.4h,  v0.h[6]		// old line1 [d3  d2  d1  d0] x kernel line 2
	fmul	v16.4h, v5.4h,  v0.h[0]		// new line0 [d3  d2  d1  d0] x kernel line 0
	fmla	v18.4h,v24.4h,  v0.h[7]
        add     x9,  x9,  x4, LSL 2
	fmla	v16.4h,v20.4h,  v0.h[1]
        add     x15, x15, x4, LSL 2
	fmla	v18.4h,v25.4h,  v1.h[0]
	fmla	v16.4h,v21.4h,  v0.h[2]

	fadd	v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d18, [x10]
        add     x10,x10,x4, LSL 1

	fmla	v16.4h, v2.4h,  v0.h[3]		// new line0 [d3  d2  d1  d0] x kernel line 1
	fmul	v18.4h, v2.4h,  v0.h[0]		// new line1 [d3  d2  d1  d0] x kernel line 0
	fmla	v16.4h,v24.4h,  v0.h[4]
	fmla	v18.4h,v24.4h,  v0.h[1]
	fmla	v16.4h,v25.4h,  v0.h[5]
	fmla	v18.4h,v25.4h,  v0.h[2]
        bne     more4_column_line_loop

more4_column_line_loop_end:
	add	x12, x12, 4			// update column counter
	and	x11, x5, 0x1
	cbnz	x11, more4_column_line_last0
        ldr     d5, [x9]   	    		// v5 = [i12 i11 i10 i1-1]
        ldr     s6, [x9, 0x8]     		// v6 = [ 0   0  i14 i13 ]
        ext     v20.8b, v5.8b,  v6.8b, 2        // v20 = [i13 i12 i11 i10]
        ext     v21.8b, v5.8b,  v6.8b, 4        // v21 = [i14 i13 i12 i11]
	fmla	v16.4h, v5.4h,  v0.h[6]		// old line0 [d3  d2  d1  d0] x kernel line 2
	fmla	v18.4h, v5.4h,  v0.h[3]		// old line1 [d3  d2  d1  d0] x kernel line 1
	fmla	v16.4h,v20.4h,  v0.h[7]
	prfm	pldl1keep, [x9, 0x40]
	fmla	v18.4h,v20.4h,  v0.h[4]
	fmla	v16.4h,v21.4h,  v1.h[0]
	fmla	v18.4h,v21.4h,  v0.h[5]

	fadd	v16.4h, v16.4h, v28.4h
	fadd	v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
	fmax	v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1
        str     d18, [x10]
        b       more4_column_start

more4_column_line_last0:
	fadd	v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]
	b	more4_column_start

more4_column_finish:
	mul	x14, x4, x5			// x14 is used as next page offset
	sub	x14, x14,x4
	lsl	x14, x14, 1

	sub	x13, x4, x12
	add	x14, x14, 10
	cmp	x13, 4
	beq	last4_column_start
	sub	x12, x12, 1
	cmp	x13, 3
	beq	last4_column_start
	sub	x12, x12, 1
	cmp	x13, 2
	beq	last4_column_start
	sub	x12, x12, 1

	// last 4 column
last4_column_start:
	add	x14, x14, 10
	add	x9, x0, x12, LSL 1	
	add	x10,x2, x12, LSL 1		// initial output pointer
	sub	x9, x9, 2			// initial input pointer
	// first line
        ldr     d2, [x9]            		// v2 = [i02 i01 i00 i0-1]
        ldr     h3, [x9, 0x8]      		// v3 = [ 0   0   0  i03 ]
        sub     x11, x5, 1
        ext     v20.8b, v2.8b, v3.8b, 2         // v20 = [i03 i02 i01 i00]
        cmp     x11, 3
        ext     v21.8b, v2.8b, v3.8b, 4         // v21 = [ 0  i03 i02 i01]
        fmul    v16.4h, v2.4h,  v0.h[3]         // line0 [d3  d2  d1  d0] x kernel line 1
        fmul    v18.4h, v2.4h,  v0.h[0]         // line1 [d3  d2  d1  d0] x kernel line 0
        fmla    v16.4h, v20.4h, v0.h[4]
	prfm	pldl1keep, [x9, x14]
        fmla    v18.4h, v20.4h, v0.h[1]
        add     x9, x9, x4, LSL 1
        fmla    v16.4h, v21.4h, v0.h[5]
        fmla    v18.4h, v21.4h, v0.h[2]
        blt     last4_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 1

	// looped 2 more lines
last4_column_line_loop:
        // line 1
        ldr     d5, [x9]   	    		// v5 = [i12 i11 i10 i1-1]
        ldr     h6, [x9, 0x8]     		// v6 = [ 0   0   0  i13 ]
        ldr     d2, [x15]      			// v2 = [i22 i21 i20 i2-1]
        ldr     h3, [x15,0x8]     		// v3 = [ 0   0   0  i23 ]
        subs    x11,x11, 1
        ext     v20.8b, v5.8b,  v6.8b, 2        // v20 = [i13 i12 i11 i10]
        ext     v21.8b, v5.8b,  v6.8b, 4        // v21 = [ 0  i13 i12 i11]
        ext     v24.8b, v2.8b,  v3.8b, 2        // v24 = [i23 i22 i21 i20]
        ext     v25.8b, v2.8b,  v3.8b, 4        // v25 = [ 0  i23 i22 i21]
	fmla	v16.4h, v5.4h,  v0.h[6]		// old line0 [d3  d2  d1  d0] x kernel line 2
	fmla	v18.4h, v5.4h,  v0.h[3]		// old line1 [d3  d2  d1  d0] x kernel line 1
	prfm	pldl1keep, [x9, x14]
	fmla	v16.4h,v20.4h,  v0.h[7]
	fmla	v18.4h,v20.4h,  v0.h[4]
	fmla	v16.4h,v21.4h,  v1.h[0]
	prfm	pldl1keep, [x15,x14]
	fmla	v18.4h,v21.4h,  v0.h[5]

	fadd	v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1

	fmla	v18.4h, v2.4h,  v0.h[6]		// old line1 [d3  d2  d1  d0] x kernel line 2
	fmul	v16.4h, v5.4h,  v0.h[0]		// new line0 [d3  d2  d1  d0] x kernel line 0
        add     x9,  x9,  x4, LSL 2
	fmla	v18.4h,v24.4h,  v0.h[7]
        add     x15, x15, x4, LSL 2
	fmla	v16.4h,v20.4h,  v0.h[1]
	fmla	v18.4h,v25.4h,  v1.h[0]
	fmla	v16.4h,v21.4h,  v0.h[2]

	fadd	v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d18, [x10]
        add     x10,x10,x4, LSL 1

	fmla	v16.4h, v2.4h,  v0.h[3]		// new line0 [d3  d2  d1  d0] x kernel line 1
	fmul	v18.4h, v2.4h,  v0.h[0]		// new line1 [d3  d2  d1  d0] x kernel line 0
	fmla	v16.4h,v24.4h,  v0.h[4]
	fmla	v18.4h,v24.4h,  v0.h[1]
	fmla	v16.4h,v25.4h,  v0.h[5]
	fmla	v18.4h,v25.4h,  v0.h[2]
        bne     last4_column_line_loop

last4_column_line_loop_end:
	and	x11, x5, 0x1
	cbnz	x11, last4_column_line_last0
        
        ldr     d5, [x9]   	    		// v5 = [i12 i11 i10 i1-1]
        ldr     h6, [x9, 0x8]     		// v6 = [ 0   0   0  i13 ]
        ext     v20.8b, v5.8b,  v6.8b, 2        // v20 = [i13 i12 i11 i10]
        ext     v21.8b, v5.8b,  v6.8b, 4        // v21 = [ 0  i13 i12 i11]
	fmla	v16.4h, v5.4h,  v0.h[6]		// old line0 [d3  d2  d1  d0] x kernel line 2
	fmla	v18.4h, v5.4h,  v0.h[3]		// old line1 [d3  d2  d1  d0] x kernel line 1
	fmla	v16.4h,v20.4h,  v0.h[7]
	prfm	pldl1keep, [x9, x14]
	fmla	v18.4h,v20.4h,  v0.h[4]
	fmla	v16.4h,v21.4h,  v1.h[0]
	fmla	v18.4h,v21.4h,  v0.h[5]

	fadd	v16.4h, v16.4h, v28.4h
	fadd	v18.4h, v18.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
	fmax	v18.4h, v18.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
	fmin	v18.4h, v18.4h, v30.4h
#endif
#endif
        str     d16, [x10]
        add     x10,x10, x4, LSL 1
        str     d18, [x10]
        b       channel_end

last4_column_line_last0:
	fadd	v16.4h, v16.4h, v28.4h
#ifdef CONV_RELU_FUSE
	fmax	v16.4h, v16.4h, v29.4h
#ifdef CONV_RELU6_FUSE
	fmin	v16.4h, v16.4h, v30.4h
#endif
#endif
        str     d16, [x10]

channel_end:
	subs	x3, x3, 1
	mul	x13,x4, x5
	lsl	x13,x13, 1
	add	x0, x0, x13
	add	x2, x2, x13
	bne	channel_loop

	ret
