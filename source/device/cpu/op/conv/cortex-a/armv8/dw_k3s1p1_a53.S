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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: xiaowei@openailab.com
 */

//
// depthwise convolution kernel size 3x3 stride 1  pad 1
// input:
//        x0     arg0  input data address
//        x1     arg1  kernel data address
//        x2     arg2  output data address
//        x3     arg3  channel number
//        x4     arg4  input width
//        x5     arg5  input height    must >=4
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
// kernel q0     xx k02 k01 k00
//        q1     xx k12 k11 k10
//        q2     xx k22 k21 k20
//
// input  q3 ~ q7
//        q8 ~ q12
//
// bias   q13
// relu   q14, q15
//
// output line0 q16 ~ s19
//        line1 q20 ~ q23
//
// temp   q24 ~ q27
//        q28 ~ q31
//

#define CONV_RELU_FUSE

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1_a53
#endif

        .section .text, "ax"
        .align 5
        .type KERNEL_NAME STT_FUNC
        .global KERNEL_NAME
        .hidden KERNEL_NAME
KERNEL_NAME:
        // context save & load parameter
	sub	sp, sp, 0x40
	stp	d8, d9, [sp]
	stp	d10,d11,[sp, 0x10]
	stp	d12,d13,[sp, 0x20]
	stp	d14,d15,[sp, 0x30]
#ifdef CONV_RELU_FUSE
	movi	d14, 0
	scvtf   s15, w7
	dup  v15.4s, v15.s[0]
#endif
channel_loop:
	movi	d13, 0
	cbz	x6, no_biases
	ld1r	{v13.4s}, [x6], 0x4
no_biases:
	mov	x9, x0				// intial channel input point
	mov	x10,x2				// intial channel output point
	mov	x12, 0				// initial column counter
	// load kernel
	ldp	q0, q1, [x1]
	ldr	s2, [x1, 0x20]
	ext	v2.16b, v1.16b, v2.16b,8
	prfm	pldl2keep, [x1, 0x40]
	ext	v1.16b, v0.16b, v1.16b,12
	add	x1, x1, 0x24

        cmp     x4, 9
        blt     first4_column_start             // if input width >= 9 use long start

first8_column_start:
    // first 8 line
        ldp     q3, q4, [x9]                    // v4 = [i07 i06 i05 i04] v3 = [i03 i02 i01 i00]
        ldr     s5, [x9, 0x20]                  // v5 = [ 0   0   0  i08]
        sub     x11, x5, 1
        ext     v28.16b,v5.16b, v3.16b, 12      // v28 = [i02 i01 i00  0 ]
        ext     v29.16b,v3.16b, v4.16b, 4       // v29 = [i04 i03 i02 i01]
        ext     v30.16b,v3.16b, v4.16b, 12      // v30 = [i06 i05 i04 i03]
        ext     v31.16b,v4.16b, v5.16b, 4       // v31 = [i08 i07 i06 i05]
        cmp     x11, 3
        fmul    v16.4s, v28.4s, v1.s[0]         // line0 [d3  d2  d1  d0]
        fmul    v17.4s, v30.4s, v1.s[0]         // line0 [d7  d6  d5  d4]
        fmul    v20.4s, v28.4s, v0.s[0]         // line1 [d3  d2  d1  d0]
        fmul    v21.4s, v30.4s, v0.s[0]         // line1 [d7  d6  d5  d4]
        fmla    v16.4s, v3.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0x30]
        fmla    v17.4s, v4.4s,  v1.s[1]
        fmla    v20.4s, v3.4s,  v0.s[1]
	prfm	pldl1keep, [x9, 0x70]
        fmla    v21.4s, v4.4s,  v0.s[1]
        fmla    v16.4s, v29.4s, v1.s[2]
	prfm	pldl1keep, [x9, 0xb0]
        fmla    v17.4s, v31.4s, v1.s[2]
        add     x9, x9, x4, LSL 2
        fmla    v20.4s, v29.4s, v0.s[2]
        fmla    v21.4s, v31.4s, v0.s[2]
        blt     first8_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 2
        // looped 2 more lines
first8_column_line_loop:
        // line 1
        ldp     q8, q9, [x9]                    // v9 = [i17 i16 i15 i14] v8 = [i13 i12 i11 i10]
        ldr     s10,[x9, 0x20]                  // v10= [ 0   0   0  i18]
        ldp     q3, q4, [x15]                   // v4 = [i27 i26 i25 i24] v3 = [i23 i22 i21 i20]
        ldr     s5, [x15, 0x20]                 // v5 = [ 0   0   0  i28]
        ext     v24.16b,v10.16b,v8.16b, 12      // v24 = [i12 i11 i10  0 ]
        ext     v25.16b,v8.16b, v9.16b, 4       // v25 = [i14 i13 i12 i11]
        ext     v26.16b,v8.16b, v9.16b, 12      // v26 = [i16 i15 i14 i13]
        ext     v27.16b,v9.16b,v10.16b, 4       // v27 = [i18 i17 i16 i15]
        ext     v28.16b,v5.16b, v3.16b, 12      // v28 = [i22 i21 i20  0 ]
        ext     v29.16b,v3.16b, v4.16b, 4       // v29 = [i24 i23 i22 i21]
        ext     v30.16b,v3.16b, v4.16b, 12      // v30 = [i26 i25 i24 i23]
        ext     v31.16b,v4.16b, v5.16b, 4       // v31 = [i28 i27 i26 i25]
        fmla    v16.4s, v24.4s, v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v17.4s, v26.4s, v2.s[0]         // old line0 [d7  d6  d5  d4]
        fmla    v20.4s, v24.4s, v1.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v21.4s, v26.4s, v1.s[0]         // old line1 [d7  d6  d5  d4]
        fmla    v16.4s, v8.4s,  v2.s[1]
	prfm	pldl1keep, [x9, 0x30]
        fmla    v17.4s, v9.4s,  v2.s[1]
        fmla    v20.4s, v8.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0x70]
        fmla    v21.4s, v9.4s,  v1.s[1]
        fmla    v16.4s, v25.4s, v2.s[2]
	prfm	pldl1keep, [x9, 0xb0]
        fmla    v17.4s, v27.4s, v2.s[2]
        add     x9, x9, x4, LSL 3
        fmla    v20.4s, v25.4s, v1.s[2]
        fmla    v21.4s, v27.4s, v1.s[2]

        fadd    v16.4s, v16.4s, v13.4s
        fadd    v17.4s, v17.4s, v13.4s
#ifdef CONV_RELU_FUSE
		cmp w7,0
		blt save_done_100f
        fmax    v16.4s, v16.4s, v14.4s
        fmax    v17.4s, v17.4s, v14.4s
		beq save_done_100f
	    fmin	v16.4s, v16.4s, v15.4s
	    fmin	v17.4s, v17.4s, v15.4s

save_done_100f:
#endif
        stp     q16, q17, [x10]
        add     x10,x10, x4, LSL 2
        fmul    v16.4s, v24.4s, v0.s[0]         // new line0 [d3  d2  d1  d0]
        fmul    v17.4s, v26.4s, v0.s[0]         // new line0 [d7  d6  d5  d4]
        fmla    v20.4s, v28.4s, v2.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v21.4s, v30.4s, v2.s[0]         // old line1 [d7  d6  d5  d4]
        fmla    v16.4s, v8.4s,  v0.s[1]
	prfm	pldl1keep, [x15, 0x30]
        fmla    v17.4s, v9.4s,  v0.s[1]
        fmla    v20.4s, v3.4s,  v2.s[1]
	prfm	pldl1keep, [x15, 0x70]
        fmla    v21.4s, v4.4s,  v2.s[1]
        fmla    v16.4s, v25.4s, v0.s[2]
	prfm	pldl1keep, [x15, 0xb0]
        fmla    v17.4s, v27.4s, v0.s[2]
        fmla    v20.4s, v29.4s, v2.s[2]
        add     x15,x15,x4, LSL 3
        fmla    v21.4s, v31.4s, v2.s[2]
        fadd    v20.4s, v20.4s, v13.4s
        fadd    v21.4s, v21.4s, v13.4s
#ifdef CONV_RELU_FUSE
		cmp w7,0
		blt save_done_101f
        fmax    v20.4s, v20.4s, v14.4s
        fmax    v21.4s, v21.4s, v14.4s
		beq save_done_101f
	fmin	v20.4s, v20.4s, v15.4s
	fmin	v21.4s, v21.4s, v15.4s
save_done_101f:

#endif
        subs    x11,x11, 1
        stp     q20, q21,[x10]
        add     x10,x10,x4, LSL 2
        fmla    v16.4s, v28.4s, v1.s[0]         // new line0 [d3  d2  d1  d0]
        fmla    v17.4s, v30.4s, v1.s[0]         // new line0 [d7  d6  d5  d4]
        fmul    v20.4s, v28.4s, v0.s[0]         // new line1 [d3  d2  d1  d0]
        fmul    v21.4s, v30.4s, v0.s[0]         // new line1 [d7  d6  d5  d4]
        fmla    v16.4s, v3.4s,  v1.s[1]
        fmla    v17.4s, v4.4s,  v1.s[1]
        fmla    v20.4s, v3.4s,  v0.s[1]
        fmla    v16.4s, v29.4s, v1.s[2]
        fmla    v17.4s, v31.4s, v1.s[2]
        fmla    v20.4s, v29.4s, v0.s[2]
        fmla    v21.4s, v4.4s,  v0.s[1]
        fmla    v21.4s, v31.4s, v0.s[2]

        bne     first8_column_line_loop
first8_column_line_loop_end:
        and     x11, x5, 0x1
        add     x12, x12, 8
        cbnz    x11, first8_column_line_last0
        // last 1 line
        ldp     q8, q9, [x9]                    // v9 = [i17 i16 i15 i14] v8 = [i13 i12 i11 i10]
        ldr     s10,[x9, 0x20]                  // v10= [ 0   0   0  i18]

        ext     v24.16b,v10.16b,v8.16b, 12      // v24 = [i12 i11 i10  0 ]
        ext     v25.16b,v8.16b, v9.16b, 4       // v25 = [i14 i13 i12 i11]
        ext     v26.16b,v8.16b, v9.16b, 12      // v26 = [i16 i15 i14 i13]
        ext     v27.16b,v9.16b,v10.16b, 4       // v27 = [i18 i17 i16 i15]
        fmla    v16.4s, v24.4s, v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v17.4s, v26.4s, v2.s[0]         // old line0 [d7  d6  d5  d4]
        fmla    v20.4s, v24.4s, v1.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v21.4s, v26.4s, v1.s[0]         // old line1 [d7  d6  d5  d4]
        fmla    v16.4s, v8.4s,  v2.s[1]
        fmla    v17.4s, v9.4s,  v2.s[1]
	prfm	pldl1keep, [x9, 0x30]
        fmla    v20.4s, v8.4s,  v1.s[1]
        fmla    v21.4s, v9.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0x70]
        fmla    v16.4s, v25.4s, v2.s[2]
        fmla    v17.4s, v27.4s, v2.s[2]
	prfm	pldl1keep, [x9, 0xb0]
        fmla    v20.4s, v25.4s, v1.s[2]
        fmla    v21.4s, v27.4s, v1.s[2]
        fadd    v16.4s, v16.4s, v13.4s
        fadd    v17.4s, v17.4s, v13.4s
        fadd    v20.4s, v20.4s, v13.4s
        fadd    v21.4s, v21.4s, v13.4s
#ifdef CONV_RELU_FUSE
		cmp w7,0
		blt save_done_102f
        fmax    v16.4s, v16.4s, v14.4s
        fmax    v17.4s, v17.4s, v14.4s
        fmax    v20.4s, v20.4s, v14.4s
        fmax    v21.4s, v21.4s, v14.4s
		beq save_done_102f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v17.4s, v17.4s, v15.4s
	fmin	v20.4s, v20.4s, v15.4s
	fmin	v21.4s, v21.4s, v15.4s
save_done_102f:
#endif
        stp     q16, q17, [x10]
        add     x10,x10, x4, LSL 2
        stp     q20, q21,[x10]
        b       first_column_finish
first8_column_line_last0:
        fadd    v16.4s, v16.4s, v13.4s
        fadd    v17.4s, v17.4s, v13.4s
#ifdef CONV_RELU_FUSE
		cmp w7,0
		blt save_done_103f
        fmax    v16.4s, v16.4s, v14.4s
        fmax    v17.4s, v17.4s, v14.4s
		beq save_done_103f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v17.4s, v17.4s, v15.4s

save_done_103f:
#endif
        stp     q16, q17, [x10]
        b       first_column_finish

    // first 4 column
	// first line
first4_column_start:
	ldr	q3, [x9]			// v3 = [i03 i02 i01 i00]
	ldr	s4, [x9, 0x10] 			// v4 = [ 0   0   0  i04]
	sub	x11, x5, 1
	cmp	x11, 3
	ext	v28.16b,v4.16b, v3.16b, 12	// v28 = [i02 i01 i00  0 ]
	ext	v29.16b,v3.16b, v4.16b, 4	// v29 = [i04 i03 i02 i01]
	fmul	v16.4s, v28.4s, v1.s[0]		// line0 [d3  d2  d1  d0]
	prfm	pldl1keep, [x9, 0x20]
	fmul	v20.4s, v28.4s, v0.s[0]		// line1 [d3  d2  d1  d0]
	prfm	pldl1keep, [x9, 0x60]
	fmla	v16.4s, v3.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0xa0]
	fmla	v20.4s, v3.4s,  v0.s[1]
	fmla	v16.4s, v29.4s, v1.s[2]
	add	x9, x9, x4, LSL 2
	fmla	v20.4s, v29.4s, v0.s[2]
	blt	first4_column_line_loop_end
	lsr	x11, x11, 1
	add	x15,x9, x4, LSL 2

	// looped 2 more lines
first4_column_line_loop:
	// line 1
	ldr	q8, [x9]			// v8 = [i13 i12 i11 i10]
	ldr	q3, [x15]			// v3 = [i23 i22 i21 i20]
	ldr	s9, [x9, 0x10]			// v9 = [ 0   0   0  i14]
	ldr	s4, [x15, 0x10]			// v4 = [ 0   0   0  i24]
	ext	v28.16b,v9.16b, v8.16b, 12	// v28 = [i12 i11 i10  0 ]
	ext	v29.16b,v8.16b, v9.16b, 4	// v29 = [i14 i13 i12 i11]
	ext	v30.16b,v4.16b, v3.16b, 12	// v30 = [i22 i21 i20  0 ]
	ext	v31.16b,v3.16b, v4.16b, 4	// v31 = [i24 i23 i22 i21]
	fmla	v16.4s, v28.4s, v2.s[0]		// old line0 [d3  d2  d1  d0]
	fmla	v20.4s, v28.4s, v1.s[0]		// old line1 [d3  d2  d1  d0]
	prfm	pldl1keep, [x9, 0x20]
	fmla	v16.4s, v8.4s,  v2.s[1]
	prfm	pldl1keep, [x9, 0x60]
	fmla	v20.4s, v8.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0xa0]
	fmla	v16.4s, v29.4s, v2.s[2]
	fmla	v20.4s, v29.4s, v1.s[2]
	add	x9, x9, x4, LSL 3
	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_104f
	fmax	v16.4s, v16.4s, v14.4s
	beq save_done_104f
	fmin	v16.4s, v16.4s, v15.4s
save_done_104f:
#endif
	str	q16, [x10]
	fmul	v16.4s, v28.4s, v0.s[0]		// new line0 [d3  d2  d1  d0]
	add	x10,x10, x4, LSL 2
	fmla	v16.4s, v8.4s,  v0.s[1]
	fmla	v16.4s, v29.4s, v0.s[2]
	// line 0
	fmla	v20.4s, v30.4s, v2.s[0]		// old line1 [d3  d2  d1  d0]
	fmla	v16.4s, v30.4s, v1.s[0]		// new line0 [d3  d2  d1  d0]
	fmla	v20.4s, v3.4s,  v2.s[1]
	prfm	pldl1keep, [x15, 0x20]
	fmla	v16.4s, v3.4s,  v1.s[1]
	prfm	pldl1keep, [x15, 0x60]
	fmla	v20.4s, v31.4s, v2.s[2]
	prfm	pldl1keep, [x15, 0xa0]
	fmla	v16.4s, v31.4s, v1.s[2]
	add	x15,x15,x4, LSL 3
	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_105f
	fmax	v20.4s, v20.4s, v14.4s
	beq save_done_105f
	fmin	v20.4s, v20.4s, v15.4s
save_done_105f:
#endif
	subs	x11,x11, 1
	str	q20, [x10]
	fmul	v20.4s, v30.4s, v0.s[0]		// new line1 [d3  d2  d1  d0]
	add	x10,x10,x4, LSL 2
	fmla	v20.4s, v3.4s,  v0.s[1]
	fmla	v20.4s, v31.4s, v0.s[2]
	bne	first4_column_line_loop

first4_column_line_loop_end:
	and	x11, x5, 0x1
	add	x12, x12, 4			// update column counter
	cbnz	x11, first4_column_line_last0
	// last 1 line
	ldr	q8, [x9]			// v8 = [i13 i12 i11 i10]
	ldr	s9, [x9, 0x10]			// v9 = [ 0   0   0  i14]
	ext	v28.16b,v9.16b, v8.16b, 12	// v28 = [i12 i11 i10  0 ]
	ext	v29.16b,v8.16b, v9.16b, 4	// v29 = [i14 i13 i12 i11]

	fmla	v16.4s, v28.4s, v2.s[0]		// old line0 [d3  d2  d1  d0]
	fmla	v20.4s, v28.4s, v1.s[0]		// old line1 [d3  d2  d1  d0]
	fmla	v16.4s, v8.4s,  v2.s[1]
	prfm	pldl1keep, [x9, 0x20]
	fmla	v20.4s, v8.4s,  v1.s[1]
	prfm	pldl1keep, [x9, 0x60]
	fmla	v16.4s, v29.4s, v2.s[2]
	prfm	pldl1keep, [x9, 0xa0]
	fmla	v20.4s, v29.4s, v1.s[2]
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_106f
	fmax	v16.4s, v16.4s, v14.4s
	fmax	v20.4s, v20.4s, v14.4s
	beq save_done_106f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v20.4s, v20.4s, v15.4s
save_done_106f:
#endif
	str	q16, [x10]
	add	x10,x10, x4, LSL 2
	str	q20, [x10]
	b	first_column_finish

first4_column_line_last0:
	// last 0 line
	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_107f
	fmax	v16.4s, v16.4s, v14.4s
	beq save_done_107f
	fmin	v16.4s, v16.4s, v15.4s

save_done_107f:
#endif
	str	q16, [x10]

first_column_finish:

    // 16 more column
more16_column_start:
	sub	x13, x4, x12
	cmp	x13, 17	// must have 5 more column to start more4 column loop
	blt	more16_column_finish
	add	x9, x0, x12, LSL 2
	add	x10,x2, x12, LSL 2		// initial output pointer
	sub	x9, x9, 4			// initial input pointer
	// first line
        ldp     q3, q4, [x9]                	// v4 = [i06 i05 i04 i03 ] v3 = [i02 i01 i00 i0-1]
	ldp	q5, q6, [x9, 0x20]		// v6 = [i0e i0d i0c i0b ] v5 = [i0a i09 i08 i07 ]
        ldr     d7, [x9, 0x40]          	// v7 = [ 0   0  i0g i0f ]
	sub	x11, x5, 1
        ext     v24.16b,v3.16b, v4.16b, 4       // v24 = [i03 i02 i01 i00]
        ext     v25.16b,v3.16b, v4.16b, 8       // v25 = [i04 i03 i02 i01]
        ext     v26.16b,v4.16b, v5.16b, 4       // v26 = [i07 i06 i05 i04]
        ext     v27.16b,v4.16b, v5.16b, 8       // v27 = [i08 i07 i06 i05]
        ext     v28.16b,v5.16b, v6.16b, 4       // v28 = [i0b i0a i09 i08]
        ext     v29.16b,v5.16b, v6.16b, 8       // v29 = [i0c i0b i0a i09]
        ext     v30.16b,v6.16b, v7.16b, 4       // v30 = [i0f i0e i0d i0c]
        ext     v31.16b,v6.16b, v7.16b, 8       // v31 = [i0g i0f i0e i0d]
	cmp	x11, 3
        fmul    v16.4s, v3.4s,  v1.s[0]         // line0 [d3  d2  d1  d0]
	fmul	v17.4s, v4.4s,  v1.s[0]		// line0 [d7  d6  d5  d4]
        fmul    v18.4s, v5.4s,  v1.s[0]         // line0 [db  da  d9  d8]
        fmul    v19.4s, v6.4s,  v1.s[0]         // line0 [df  de  dd  dc]
        fmla    v16.4s, v24.4s, v1.s[1]
	fmla	v17.4s, v26.4s, v1.s[1]
        fmla    v18.4s, v28.4s, v1.s[1]
        fmla    v19.4s, v30.4s, v1.s[1]
	prfm	pldl1keep, [x9, 0x50]
        fmla    v16.4s, v25.4s, v1.s[2]
	prfm	pldl1keep, [x9, 0x90]
	fmla	v17.4s, v27.4s, v1.s[2]
        fmla    v18.4s, v29.4s, v1.s[2]
        fmla    v19.4s, v31.4s, v1.s[2]
        add     x9, x9, x4, LSL 2
        fmul    v20.4s, v3.4s,  v0.s[0]         // line1 [d3  d2  d1  d0]
	fmul	v21.4s, v4.4s,  v0.s[0]		// line1 [d7  d6  d5  d4]
        fmul    v22.4s, v5.4s,  v0.s[0]         // line1 [db  da  d9  d8]
	fmul	v23.4s, v6.4s,  v0.s[0]		// line1 [df  de  dd  dc]
        fmla    v20.4s, v24.4s, v0.s[1]
	fmla	v21.4s, v26.4s, v0.s[1]
        fmla    v22.4s, v28.4s, v0.s[1]
	fmla	v23.4s, v30.4s, v0.s[1]
        fmla    v20.4s, v25.4s, v0.s[2]
	fmla	v21.4s, v27.4s, v0.s[2]
        fmla    v22.4s, v29.4s, v0.s[2]
	fmla	v23.4s, v31.4s, v0.s[2]
        blt     more16_column_line_loop_end
        lsr     x11, x11, 1
        add     x15, x9, x4, LSL 2
	// looped 2 more lines
more16_column_line_loop:
        // line 1
        ldp     q8, q9, [x9]                	// v9 = [i16 i15 i14 i13 ] v8 = [i12 i11 i10 i1-1]
	ldp	q10,q11,[x9, 0x20]		// v11= [i1e i1d i1c i1b ] v10= [i1a i19 i18 i17 ]
        ldr     d12, [x9, 0x40]          	// v12= [ 0   0  i1g i1f ]
        ldp     q3, q4, [x15]                	// v4 = [i26 i25 i24 i23 ] v3 = [i22 i21 i20 i2-1]
	ldp	q5, q6, [x15, 0x20]		// v6 = [i2e i2d i2c i2b ] v5 = [i2a i29 i28 i27 ]
        ldr     d7, [x15, 0x40]          	// v7 = [ 0   0  i0g i0f ]
        ext     v24.16b,v8.16b, v9.16b, 4       // v24 = [i13 i12 i11 i10]
        ext     v25.16b,v8.16b, v9.16b, 8       // v25 = [i14 i13 i12 i11]
        ext     v26.16b,v9.16b, v10.16b,4       // v26 = [i17 i16 i15 i14]
        ext     v27.16b,v9.16b, v10.16b,8       // v27 = [i18 i17 i16 i15]
        ext     v28.16b,v10.16b,v11.16b,4       // v28 = [i1b i1a i19 i18]
        ext     v29.16b,v10.16b,v11.16b,8       // v29 = [i1c i1b i1a i19]
        ext     v30.16b,v11.16b,v12.16b,4       // v30 = [i1f i1e i1d i1c]
        ext     v31.16b,v11.16b,v12.16b,8       // v31 = [i1g i1f i1e i1d]
        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
	fmla	v17.4s, v9.4s,  v2.s[0]		// old line0 [d7  d6  d5  d4]
        fmla    v18.4s, v10.4s, v2.s[0]         // old line0 [db  da  d9  d8]
        fmla    v19.4s, v11.4s, v2.s[0]         // old line0 [df  de  dd  dc]
        fmla    v16.4s, v24.4s, v2.s[1]
	fmla	v17.4s, v26.4s, v2.s[1]
	prfm	pldl1keep, [x9, 0x50]
        fmla    v18.4s, v28.4s, v2.s[1]
        fmla    v19.4s, v30.4s, v2.s[1]
	prfm	pldl1keep, [x9, 0x90]
        fmla    v16.4s, v25.4s, v2.s[2]
	fmla	v17.4s, v27.4s, v2.s[2]
        add     x9, x9, x4, LSL 3
        fmla    v18.4s, v29.4s, v2.s[2]
        fmla    v19.4s, v31.4s, v2.s[2]
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v17.4s, v17.4s, v13.4s
	fadd	v18.4s, v18.4s, v13.4s
	fadd	v19.4s, v19.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_108f
	fmax	v16.4s, v16.4s, v14.4s
	fmax	v17.4s, v17.4s, v14.4s
	fmax	v18.4s, v18.4s, v14.4s
	fmax	v19.4s, v19.4s, v14.4s
	beq save_done_108f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v17.4s, v17.4s, v15.4s
	fmin	v18.4s, v18.4s, v15.4s
	fmin	v19.4s, v19.4s, v15.4s
save_done_108f:
#endif
        fmla    v20.4s, v8.4s,  v1.s[0]         // line1 [d3  d2  d1  d0]
	fmla	v21.4s, v9.4s,  v1.s[0]		// line1 [d7  d6  d5  d4]
        fmla    v22.4s, v10.4s, v1.s[0]         // line1 [db  da  d9  d8]
	fmla	v23.4s, v11.4s, v1.s[0]		// line1 [df  de  dd  dc]
        fmla    v20.4s, v24.4s, v1.s[1]
	fmla	v21.4s, v26.4s, v1.s[1]
	prfm	pldl1keep, [x15, 0x50]
        fmla    v22.4s, v28.4s, v1.s[1]
	fmla	v23.4s, v30.4s, v1.s[1]
	prfm	pldl1keep, [x15, 0x90]
        fmla    v20.4s, v25.4s, v1.s[2]
	fmla	v21.4s, v27.4s, v1.s[2]
        add     x15, x15, x4, LSL 3
        fmla    v22.4s, v29.4s, v1.s[2]
	fmla	v23.4s, v31.4s, v1.s[2]
        stp     q16, q17, [x10]
        stp     q18, q19, [x10, 0x20]
        add     x10,x10, x4, LSL 2
        fmul    v16.4s, v8.4s,  v0.s[0]         // new line0 [d3  d2  d1  d0]
	fmul	v17.4s, v9.4s,  v0.s[0]		// new line0 [d7  d6  d5  d4]
        fmul    v18.4s, v10.4s, v0.s[0]         // new line0 [db  da  d9  d8]
        fmul    v19.4s, v11.4s, v0.s[0]         // new line0 [df  de  dd  dc]
        fmla    v16.4s, v24.4s, v0.s[1]
	fmla	v17.4s, v26.4s, v0.s[1]
        fmla    v18.4s, v28.4s, v0.s[1]
        fmla    v19.4s, v30.4s, v0.s[1]
        fmla    v16.4s, v25.4s, v0.s[2]
	fmla	v17.4s, v27.4s, v0.s[2]
        fmla    v18.4s, v29.4s, v0.s[2]
        fmla    v19.4s, v31.4s, v0.s[2]
        // line 0
        ext     v24.16b,v3.16b, v4.16b, 4       // v24 = [i23 i22 i21 i20]
        ext     v25.16b,v3.16b, v4.16b, 8       // v25 = [i24 i23 i22 i21]
        ext     v26.16b,v4.16b, v5.16b, 4       // v26 = [i27 i26 i25 i24]
        ext     v27.16b,v4.16b, v5.16b, 8       // v27 = [i28 i27 i26 i25]
        ext     v28.16b,v5.16b, v6.16b, 4       // v28 = [i2b i2a i29 i28]
        ext     v29.16b,v5.16b, v6.16b, 8       // v29 = [i2c i2b i2a i29]
        ext     v30.16b,v6.16b, v7.16b, 4       // v30 = [i2f i2e i2d i2c]
        ext     v31.16b,v6.16b, v7.16b, 8       // v31 = [i2g i2f i2e i2d]
        fmla    v20.4s, v3.4s,  v2.s[0]         // old line1 [d3  d2  d1  d0]
	fmla	v21.4s, v4.4s,  v2.s[0]		// old line1 [d7  d6  d5  d4]
        fmla    v22.4s, v5.4s,  v2.s[0]         // old line1 [db  da  d9  d8]
	fmla	v23.4s, v6.4s,  v2.s[0]		// old line1 [df  de  dd  dc]
        fmla    v20.4s, v24.4s, v2.s[1]
	fmla	v21.4s, v26.4s, v2.s[1]
        fmla    v22.4s, v28.4s, v2.s[1]
	fmla	v23.4s, v30.4s, v2.s[1]
        fmla    v20.4s, v25.4s, v2.s[2]
	fmla	v21.4s, v27.4s, v2.s[2]
        fmla    v22.4s, v29.4s, v2.s[2]
	fmla	v23.4s, v31.4s, v2.s[2]
	fadd	v20.4s, v20.4s, v13.4s
	fadd	v21.4s, v21.4s, v13.4s
	fadd	v22.4s, v22.4s, v13.4s
	fadd	v23.4s, v23.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_109f
	fmax	v20.4s, v20.4s, v14.4s
	fmax	v21.4s, v21.4s, v14.4s
	fmax	v22.4s, v22.4s, v14.4s
	fmax	v23.4s, v23.4s, v14.4s
	beq save_done_109f
	fmin	v20.4s, v20.4s, v15.4s
	fmin	v21.4s, v21.4s, v15.4s
	fmin	v22.4s, v22.4s, v15.4s
	fmin	v23.4s, v23.4s, v15.4s

save_done_109f:
#endif
        subs    x11,x11, 1
        fmla    v16.4s, v3.4s,  v1.s[0]         // new line0 [d3  d2  d1  d0]
	fmla	v17.4s, v4.4s,  v1.s[0]		// new line0 [d7  d6  d5  d4]
        fmla    v18.4s, v5.4s,  v1.s[0]         // new line0 [db  da  d9  d8]
        fmla    v19.4s, v6.4s,  v1.s[0]         // new line0 [df  de  dd  dc]
        fmla    v16.4s, v24.4s, v1.s[1]
	fmla	v17.4s, v26.4s, v1.s[1]
        fmla    v18.4s, v28.4s, v1.s[1]
        fmla    v19.4s, v30.4s, v1.s[1]
        fmla    v16.4s, v25.4s, v1.s[2]
	fmla	v17.4s, v27.4s, v1.s[2]
        fmla    v18.4s, v29.4s, v1.s[2]
        fmla    v19.4s, v31.4s, v1.s[2]
        stp     q20, q21, [x10]
	stp	q22, q23, [x10, 0x20]
        add     x10,x10,x4, LSL 2
        fmul    v20.4s, v3.4s,  v0.s[0]         // new line1 [d3  d2  d1  d0]
	fmul	v21.4s, v4.4s,  v0.s[0]		// new line1 [d7  d6  d5  d4]
        fmul    v22.4s, v5.4s,  v0.s[0]         // new line1 [db  da  d9  d8]
	fmul	v23.4s, v6.4s,  v0.s[0]		// new line1 [df  de  dd  dc]
        fmla    v20.4s, v24.4s, v0.s[1]
	fmla	v21.4s, v26.4s, v0.s[1]
        fmla    v22.4s, v28.4s, v0.s[1]
	fmla	v23.4s, v30.4s, v0.s[1]
        fmla    v20.4s, v25.4s, v0.s[2]
	fmla	v21.4s, v27.4s, v0.s[2]
        fmla    v22.4s, v29.4s, v0.s[2]
	fmla	v23.4s, v31.4s, v0.s[2]
        bne     more16_column_line_loop

more16_column_line_loop_end:
	add	x12, x12, 16			// update column counter
	and	x11, x5, 0x1
	cbnz	x11, more16_column_line_last0

        // line 1
        ldp     q8, q9, [x9]                	// v9 = [i16 i15 i14 i13 ] v8 = [i12 i11 i10 i1-1]
	ldp	q10,q11,[x9, 0x20]		// v11= [i1e i1d i1c i1b ] v10= [i1a i19 i18 i17 ]
        ldr     d12, [x9, 0x40]          	// v12= [ 0   0  i1g i1f ]
        ext     v24.16b,v8.16b, v9.16b, 4       // v24 = [i13 i12 i11 i10]
        ext     v25.16b,v8.16b, v9.16b, 8       // v25 = [i14 i13 i12 i11]
        ext     v26.16b,v9.16b, v10.16b,4       // v26 = [i17 i16 i15 i14]
        ext     v27.16b,v9.16b, v10.16b,8       // v27 = [i18 i17 i16 i15]
        ext     v28.16b,v10.16b,v11.16b,4       // v28 = [i1b i1a i19 i18]
        ext     v29.16b,v10.16b,v11.16b,8       // v29 = [i1c i1b i1a i19]
        ext     v30.16b,v11.16b,v12.16b,4       // v30 = [i1f i1e i1d i1c]
        ext     v31.16b,v11.16b,v12.16b,8       // v31 = [i1g i1f i1e i1d]
        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
	fmla	v17.4s, v9.4s,  v2.s[0]		// old line0 [d7  d6  d5  d4]
        fmla    v18.4s, v10.4s, v2.s[0]         // old line0 [db  da  d9  d8]
        fmla    v19.4s, v11.4s, v2.s[0]         // old line0 [df  de  dd  dc]
        fmla    v16.4s, v24.4s, v2.s[1]
	fmla	v17.4s, v26.4s, v2.s[1]
	prfm	pldl1keep, [x9, 0x50]
        fmla    v18.4s, v28.4s, v2.s[1]
        fmla    v19.4s, v30.4s, v2.s[1]
        fmla    v16.4s, v25.4s, v2.s[2]
	prfm	pldl1keep, [x9, 0x90]
	fmla	v17.4s, v27.4s, v2.s[2]
        fmla    v18.4s, v29.4s, v2.s[2]
        add     x9, x9, x4, LSL 2
        fmla    v19.4s, v31.4s, v2.s[2]
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v17.4s, v17.4s, v13.4s
	fadd	v18.4s, v18.4s, v13.4s
	fadd	v19.4s, v19.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_110f
	fmax	v16.4s, v16.4s, v14.4s
	fmax	v17.4s, v17.4s, v14.4s
	fmax	v18.4s, v18.4s, v14.4s
	fmax	v19.4s, v19.4s, v14.4s
	beq save_done_110f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v17.4s, v17.4s, v15.4s
	fmin	v18.4s, v18.4s, v15.4s
	fmin	v19.4s, v19.4s, v15.4s

save_done_110f:
#endif
        fmla    v20.4s, v8.4s,  v1.s[0]         // line1 [d3  d2  d1  d0]
	fmla	v21.4s, v9.4s,  v1.s[0]		// line1 [d7  d6  d5  d4]
        fmla    v22.4s, v10.4s, v1.s[0]         // line1 [db  da  d9  d8]
	fmla	v23.4s, v11.4s, v1.s[0]		// line1 [df  de  dd  dc]
        fmla    v20.4s, v24.4s, v1.s[1]
	fmla	v21.4s, v26.4s, v1.s[1]
        fmla    v22.4s, v28.4s, v1.s[1]
	fmla	v23.4s, v30.4s, v1.s[1]
        fmla    v20.4s, v25.4s, v1.s[2]
	fmla	v21.4s, v27.4s, v1.s[2]
        fmla    v22.4s, v29.4s, v1.s[2]
	fmla	v23.4s, v31.4s, v1.s[2]
        stp     q16, q17, [x10]
        stp     q18, q19, [x10, 0x20]
        add     x10,x10, x4, LSL 2
	fadd	v20.4s, v20.4s, v13.4s
	fadd	v21.4s, v21.4s, v13.4s
	fadd	v22.4s, v22.4s, v13.4s
	fadd	v23.4s, v23.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_111f
	fmax	v20.4s, v20.4s, v14.4s
	fmax	v21.4s, v21.4s, v14.4s
	fmax	v22.4s, v22.4s, v14.4s
	fmax	v23.4s, v23.4s, v14.4s
	beq save_done_111f
	fmin	v20.4s, v20.4s, v15.4s
	fmin	v21.4s, v21.4s, v15.4s
	fmin	v22.4s, v22.4s, v15.4s
	fmin	v23.4s, v23.4s, v15.4s
save_done_111f:
#endif
        stp     q20, q21, [x10]
	stp	q22, q23, [x10, 0x20]
        b       more16_column_start

more16_column_line_last0:
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v17.4s, v17.4s, v13.4s
	fadd	v18.4s, v18.4s, v13.4s
	fadd	v19.4s, v19.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_112f
	fmax	v16.4s, v16.4s, v14.4s
	fmax	v17.4s, v17.4s, v14.4s
	fmax	v18.4s, v18.4s, v14.4s
	fmax	v19.4s, v19.4s, v14.4s
	beq save_done_112f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v17.4s, v17.4s, v15.4s
	fmin	v18.4s, v18.4s, v15.4s
	fmin	v19.4s, v19.4s, v15.4s
save_done_112f:
#endif
        stp     q16, q17, [x10]
        stp     q18, q19, [x10, 0x20]
	b	more16_column_start

more16_column_finish:

    // 4 more column
more4_column_start:
	sub	x13, x4, x12
	cmp	x13, 5	// must have 5 more column to start more4 column loop
	blt	more4_column_finish
	add	x9, x0, x12, LSL 2
	add	x10,x2, x12, LSL 2		// initial output pointer
	sub	x9, x9, 4			// initial input pointer
	// first line
        ldr     q3, [x9]                	// v3 = [i02 i01 i00 i0-1]
        ldr     d4, [x9, 0x10]          	// v4 = [ 0   0  i04 i03 ]
	sub	x11, x5, 1
	cmp	x11, 3
        ext     v28.16b,v3.16b, v4.16b, 4       // v28 = [i03 i02 i01 i00]
        ext     v29.16b,v3.16b, v4.16b, 8       // v29 = [i04 i03 i02 i01]
        fmul    v16.4s, v3.4s,  v1.s[0]         // line0 [d3  d2  d1  d0]
        fmul    v20.4s, v3.4s,  v0.s[0]         // line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v1.s[1]
	prfm	pldl1keep, [x9, 0x20]
        fmla    v20.4s, v28.4s, v0.s[1]
	prfm	pldl1keep, [x9, 0x60]
        fmla    v16.4s, v29.4s, v1.s[2]
        add     x9, x9, x4, LSL 2
        fmla    v20.4s, v29.4s, v0.s[2]
        blt     more4_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 2

	// looped 2 more lines
more4_column_line_loop:
        // line 1
        ldr     q8, [x9]                        // v8 = [i12 i11 i10 i1-1]
        ldr     d9, [x9, 0x10]                  // v9 = [ 0   0  i14 i13 ]
        ldr     q3, [x15]                        // v3 = [i22 i21 i20 i2-1]
        ldr     d4, [x15, 0x10]                  // v4 = [ 0   0  i24 i23 ]
        ext     v28.16b,v8.16b, v9.16b, 4       // v28 = [i13 ii2 i11 i10]
        ext     v29.16b,v8.16b, v9.16b, 8       // v29 = [i14 i13 i12 i11]
        ext     v30.16b,v3.16b, v4.16b, 4       // v30 = [i23 i22 i21 i20]
        ext     v31.16b,v3.16b, v4.16b, 8       // v31 = [i24 i23 i22 i21]
        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v20.4s, v8.4s,  v1.s[0]         // old line1 [d3  d2  d1  d0]
	prfm	pldl1keep, [x9, 0x20]
        fmla    v16.4s, v28.4s, v2.s[1]
	prfm	pldl1keep, [x9, 0x60]
        fmla    v20.4s, v28.4s, v1.s[1]
        add     x9, x9, x4, LSL 3
        fmla    v16.4s, v29.4s, v2.s[2]
        fmla    v20.4s, v29.4s, v1.s[2]
	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_113f
	fmax	v16.4s, v16.4s, v14.4s
	beq save_done_113f
	fmin	v16.4s, v16.4s, v15.4s
save_done_113f:

#endif
        str     q16, [x10]
        add     x10,x10, x4, LSL 2
        fmul    v16.4s, v8.4s,  v0.s[0]         // new line0 [d3  d2  d1  d0]
        fmla    v20.4s, v3.4s,  v2.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v0.s[1]
	prfm	pldl1keep, [x15, 0x20]
        fmla    v20.4s, v30.4s, v2.s[1]
	prfm	pldl1keep, [x15, 0x60]
        fmla    v16.4s, v29.4s, v0.s[2]
        add     x15,x15,x4, LSL 3
        fmla    v20.4s, v31.4s, v2.s[2]

	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_114f
	fmax	v20.4s, v20.4s, v14.4s
	beq save_done_114f
	fmin	v20.4s, v20.4s, v15.4s
save_done_114f:
#endif
        subs    x11,x11, 1
        str     q20, [x10]
        add     x10,x10,x4, LSL 2
        fmla    v16.4s, v3.4s,  v1.s[0]         // new line0 [d3  d2  d1  d0]
        fmul    v20.4s, v3.4s,  v0.s[0]         // new line1 [d3  d2  d1  d0]
        fmla    v16.4s, v30.4s, v1.s[1]
        fmla    v20.4s, v30.4s, v0.s[1]
        fmla    v16.4s, v31.4s, v1.s[2]
        fmla    v20.4s, v31.4s, v0.s[2]
        bne     more4_column_line_loop

more4_column_line_loop_end:
	add	x12, x12, 4			// update column counter
	and	x11, x5, 0x1
	cbnz	x11, more4_column_line_last0

        // line 1
        ldr     q8, [x9]                        // v8 = [i12 i11 i10 i1-1]
        ldr     d9, [x9, 0x10]                  // v9 = [ 0   0  i14 i13 ]
        ext     v28.16b,v8.16b, v9.16b, 4       // v28 = [i13 ii2 i11 i10]
        ext     v29.16b,v8.16b, v9.16b, 8       // v29 = [i14 i13 i12 i11]
        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v20.4s, v8.4s,  v1.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v2.s[1]
	prfm	pldl1keep, [x9, 0x20]
        fmla    v20.4s, v28.4s, v1.s[1]
        fmla    v16.4s, v29.4s, v2.s[2]
	prfm	pldl1keep, [x9, 0x60]
        fmla    v20.4s, v29.4s, v1.s[2]
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_115f

	fmax	v16.4s, v16.4s, v14.4s
	fmax	v20.4s, v20.4s, v14.4s

	beq save_done_115f

	fmin	v16.4s, v16.4s, v15.4s
	fmin	v20.4s, v20.4s, v15.4s

save_done_115f:

#endif
        str     q16, [x10]
        add     x10,x10, x4, LSL 2
        str     q20, [x10]
        b       more4_column_start

more4_column_line_last0:
	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
    cmp w7,0
	blt save_done_116f
	fmax	v16.4s, v16.4s, v14.4s

	beq save_done_116f
	fmin	v16.4s, v16.4s, v15.4s

save_done_116f:
#endif
        str     q16, [x10]
	b	more4_column_start

more4_column_finish:
	mul	x14, x4, x5			// x14 is used as next page offset
	sub	x14, x14,x4
	lsl	x14, x14, 2

	sub	x13, x4, x12

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
	add	x14, x14, 20
	add	x9, x0, x12, LSL 2
	add	x10,x2, x12, LSL 2		// initial output pointer
	sub	x9, x9, 4			// initial input pointer
	// first line
        ldr     q3, [x9]                	// v3 = [i02 i01 i00 i0-1]
        ldr     s4, [x9, 0x10]          	// v4 = [ 0   0   0  i03 ]
	sub	x11, x5, 1
	cmp	x11, 3
        ext     v28.16b,v3.16b, v4.16b, 4       // v28 = [i03 i02 i01 i00]
        ext     v29.16b,v3.16b, v4.16b, 8       // v29 = [ 0  i03 i02 i01]
        fmul    v16.4s, v3.4s,  v1.s[0]         // line0 [d3  d2  d1  d0]
        fmul    v20.4s, v3.4s,  v0.s[0]         // line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v1.s[1]
	prfm	pldl1keep, [x9, x14]
        fmla    v20.4s, v28.4s, v0.s[1]
        add     x9, x9, x4, LSL 2
        fmla    v16.4s, v29.4s, v1.s[2]
        fmla    v20.4s, v29.4s, v0.s[2]
        blt     last4_column_line_loop_end
        lsr     x11, x11, 1
        add     x15,x9, x4, LSL 2

	// looped 2 more lines
last4_column_line_loop:
        // line 1
        ldr     q8, [x9]                        // v8 = [i12 i11 i10 i1-1]
        ldr     s9, [x9, 0x10]                  // v9 = [ 0   0   0  i13 ]
        ldr     q3, [x15]                        // v3 = [i22 i21 i20 i2-1]
        ldr     s4, [x15, 0x10]                  // v4 = [ 0   0  i24 i23 ]
        ext     v28.16b,v8.16b, v9.16b, 4       // v28 = [i13 ii2 i11 i10]
        ext     v29.16b,v8.16b, v9.16b, 8       // v29 = [ 0  i13 i12 i11]
        ext     v30.16b,v3.16b, v4.16b, 4       // v30 = [i23 i22 i21 i20]
        ext     v31.16b,v3.16b, v4.16b, 8       // v31 = [i24 i23 i22 i21]

        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v20.4s, v8.4s,  v1.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v2.s[1]
	prfm	pldl1keep, [x9, x14]
        fmla    v20.4s, v28.4s, v1.s[1]
	prfm	pldl1keep, [x15, x14]
        fmla    v16.4s, v29.4s, v2.s[2]
        add     x9, x9, x4, LSL 3
        fmla    v20.4s, v29.4s, v1.s[2]
        add     x15, x15, x4, LSL 3

	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_117f
	fmax	v16.4s, v16.4s, v14.4s
	beq save_done_117f
	fmin	v16.4s, v16.4s, v15.4s
save_done_117f:
#endif
        str     q16, [x10]
        add     x10,x10,x4, LSL 2
        // line 0
        fmul    v16.4s, v8.4s,  v0.s[0]         // new line0 [d3  d2  d1  d0]
        fmla    v20.4s, v3.4s,  v2.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v0.s[1]
        fmla    v20.4s, v30.4s, v2.s[1]
        fmla    v16.4s, v29.4s, v0.s[2]
        fmla    v20.4s, v31.4s, v2.s[2]
	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt     save_done_118f
	fmax	v20.4s, v20.4s, v14.4s
	beq     save_done_118f
	fmin	v20.4s, v20.4s, v15.4s
save_done_118f:
#endif
        subs    x11,x11, 1
        str     q20, [x10]
        add     x10,x10,x4, LSL 2
        fmla    v16.4s, v3.4s,  v1.s[0]         // new line0 [d3  d2  d1  d0]
        fmul    v20.4s, v3.4s,  v0.s[0]         // new line1 [d3  d2  d1  d0]
        fmla    v16.4s, v30.4s, v1.s[1]
        fmla    v20.4s, v30.4s, v0.s[1]
        fmla    v16.4s, v31.4s, v1.s[2]
        fmla    v20.4s, v31.4s, v0.s[2]
        bne     last4_column_line_loop

last4_column_line_loop_end:
	and	x11, x5, 0x1
	cbnz	x11, last4_column_line_last0
        // line 1
        ldr     q8, [x9]                        // v8 = [i12 i11 i10 i1-1]
        ldr     s9, [x9, 0x10]                  // v9 = [ 0   0   0  i13 ]
        ext     v28.16b,v8.16b, v9.16b, 4       // v28 = [i13 ii2 i11 i10]
        ext     v29.16b,v8.16b, v9.16b, 8       // v29 = [ 0  i13 i12 i11]
        fmla    v16.4s, v8.4s,  v2.s[0]         // old line0 [d3  d2  d1  d0]
        fmla    v20.4s, v8.4s,  v1.s[0]         // old line1 [d3  d2  d1  d0]
        fmla    v16.4s, v28.4s, v2.s[1]
        fmla    v20.4s, v28.4s, v1.s[1]
	prfm	pldl1keep, [x9, x14]
        fmla    v16.4s, v29.4s, v2.s[2]
        fmla    v20.4s, v29.4s, v1.s[2]
	fadd	v16.4s, v16.4s, v13.4s
	fadd	v20.4s, v20.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_119f
	fmax	v16.4s, v16.4s, v14.4s
	fmax	v20.4s, v20.4s, v14.4s
	beq save_done_119f
	fmin	v16.4s, v16.4s, v15.4s
	fmin	v20.4s, v20.4s, v15.4s
save_done_119f:
#endif
        str     q16, [x10]
        add     x10,x10, x4, LSL 2
        str     q20, [x10]
        b       channel_end

last4_column_line_last0:
	fadd	v16.4s, v16.4s, v13.4s
#ifdef CONV_RELU_FUSE
	cmp w7,0
	blt save_done_120f
	fmax	v16.4s, v16.4s, v14.4s
	beq save_done_120f
	fmin	v16.4s, v16.4s, v15.4s
save_done_120f:
#endif
        str     q16, [x10]

channel_end:
	subs	x3, x3, 1
	mul	x13, x4, x5
	lsl	x13, x13, 2
	add	x0, x0, x13
	add	x2, x2, x13
	bne	channel_loop

	ldp	d8, d9, [sp]
	ldp	d10,d11,[sp, 0x10]
	ldp	d12,d13,[sp, 0x20]
	ldp	d14,d15,[sp, 0x30]
	add	sp, sp, 0x40

	ret
