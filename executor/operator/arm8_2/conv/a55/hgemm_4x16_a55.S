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
// 4*16 half precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  kf |     |  b0  b1  ..  bf |         | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  .   bf |         | i3k0 i3k1 .. i3kf |
//    --              --      --               --     --               --         --                 --
//      input 4 x p             kernel p x 16            biases 4 x 16                 output 4 x 16           p = kernel size
//
//
// optimised for Cortex-A55 pipeline 52 cycle per loop (4*16*4 dot product)
//
//
// input: 
//         x0 arg0  biases address {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15}  nullptr means no biases 
//         x1 arg1  input  address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x2 arg2  kernel address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],...}
//         x3 arg3  kernel size
//         x4 arg4  output address 
//                  indirect save: output {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3],i[0-3]k[4]..}
//                    direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                 output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                 output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                 ...
//                                 output + ouput_xy * 15 : {i0k15 i1k15 i2k15 i3k15}
//         x5 arg5  output xy
//         x6 arg6  fused_relu flag     relu layers is integrated after convolution
//
// output: no
//
// register definition
// x0        biases start address
// x1        input start address
// x2        kernel start address
// x3        kernal size 
// x4        output start address
// x5        output_x * output_y
// x6        fused_relu flag
//
// v0   8h input  data {i31  i21  i11  i01  i30  i20  i10  i00}
// v1   8h input  data {i33  i23  i13  i03  i32  i22  i12  i02}
// v2-3  not used
// v4   8h kernel data {k7 | k6 | k5 | k4 | k3 | k2 | k1 | k0}[0] | [2]
// v5   8h kernel data {kf | ke | kd | kc | kb | ka | k9 | k8}[0] | [2]
// V6   8h kernel data {k7 | k6 | k5 | k4 | k3 | k2 | k1 | k0}[1] | [3]
// V7   8h kernel data {kf | ke | kd | kc | kb | ka | k9 | k8}[1] | [3]
// v8~23 not used
// v24  8h dot product {i0k7, i0k6, i0k5, i0k4, i0k3, i0k2, i0k1, i0k0}
// v25  8h dot product {i1k7, i1k6, i1k5, i1k4, i1k3, i1k2, i1k1, i1k0}
// v26  8h dot product {i2k7, i2k6, i2k5, i2k4, i2k3, i2k2, i2k1, i2k0}
// v27  8h dot product {i3k7, i3k6, i3k5, i3k4, i3k3, i3k2, i3k1, i3k0}
// v28  8h dot product {i0kf, i0ke, i0kd, i0kc, i0kb, i0ka, i0k9, i0k8}
// v29  8h dot product {i1kf, i1ke, i1kd, i1kc, i1kb, i1ka, i1k9, i1k8}
// v30  8h dot product {i2kf, i2ke, i2kd, i2kc, i2kb, i2ka, i2k9, i2k8}
// v31  8h dot product {i3kf, i3ke, i3kd, i3kc, i3kb, i3ka, i3k9, i3k8}

        .section .text,"ax"
        .align 5

        .type hgemm_4x16_a55 STT_FUNC
        .global hgemm_4x16_a55
hgemm_4x16_a55:
	// biases_initial
	cmp	x3, 0x4
	cbz	x0, none_biases
	ldp	q24, q28, [x0]
	b	convolution_start

none_biases:
	movi	d24, 0x0
	movi	d28, 0x0

convolution_start:
	and	x10,x3, 0x3
	lsl	x5, x5, 1			// x5  = output_xy
	mov	v25.16b, v24.16b
	add	x11, x4, x5			// x11 = output + ouput_xy
	mov	v26.16b, v24.16b
	add	x12, x4, x5, LSL 1		// x12 = output + ouput_xy * 2
	mov	v27.16b, v24.16b
	add	x15, x4, x5, LSL 2		// x15 = output + ouput_xy * 4
	mov	v29.16b, v28.16b
	add	x13,x11, x5, LSL 1		// x13 = output + ouput_xy * 3
	mov	v30.16b, v28.16b
	add	x16,x11, x5, LSL 2		// x16 = output + ouput_xy * 5
	mov	v31.16b, v28.16b
	add	x17,x12, x5, LSL 2		// x17 = output + ouput_xy * 6

	b.lt	loop4_end
	lsr	x9, x3, 0x2

// main loop     each loop generate dot prodcut for 4x16x4SP
loop4:  
	ldnp	q0, q1, [x1]			// q0=i[3-0][1-0] q1=i[3-0][3-2]
	ldp	q4, q5, [x2]	 		// q4=k[7-0][0] q5=k[f-8][0] 
	ldp	q6, q7, [x2, 0x20] 		// q6=k[7-0][1] q7=k[f-8][1] 
	subs	x9, x9, 0x1
	fmla	v24.8h, v4.8h,  v0.h[0]		// i[0]k[7-0]
	fmla	v25.8h, v4.8h,  v0.h[1]		// i[1]k[7-0]
	fmla	v26.8h, v4.8h,  v0.h[2]		// i[2]k[7-0]
	fmla	v27.8h, v4.8h,  v0.h[3]		// i[3]k[7-0]
	prfm	pldl1strm, [x1, 0xe0]
	fmla	v28.8h, v5.8h,  v0.h[0]		// i[0]k[f-8]
	fmla	v29.8h, v5.8h,  v0.h[1]		// i[1]k[f-8]
	add	x1, x1, 0x20
	fmla	v30.8h, v5.8h,  v0.h[2]		// i[2]k[f-8]
	fmla	v31.8h, v5.8h,  v0.h[3]		// i[3]k[f-8]
	
	fmla	v24.8h, v6.8h,  v0.h[4]		// i[0]k[7-0]
	fmla	v25.8h, v6.8h,  v0.h[5]		// i[1]k[7-0]
	fmla	v26.8h, v6.8h,  v0.h[6]		// i[2]k[7-0]
	fmla	v27.8h, v6.8h,  v0.h[7]		// i[3]k[7-0]
	fmla	v28.8h, v7.8h,  v0.h[4]		// i[0]k[f-8]
	fmla	v29.8h, v7.8h,  v0.h[5]		// i[1]k[f-8]
	fmla	v30.8h, v7.8h,  v0.h[6]		// i[2]k[f-8]
	fmla	v31.8h, v7.8h,  v0.h[7]		// i[3]k[f-8]

	ldp	q4, q5, [x2, 0x40]		// q4=k[7-0][2] q5=k[f-8][2]
	ldp	q6, q7, [x2, 0x60]		// q6=k[7-0][3] q7=k[f-8][3]
	fmla	v24.8h, v4.8h,  v1.h[0]		// i[0]k[7-0]
	fmla	v25.8h, v4.8h,  v1.h[1]		// i[1]k[7-0]
	add	x2, x2, 0x80
	fmla	v26.8h, v4.8h,  v1.h[2]		// i[2]k[7-0]
	fmla	v27.8h, v4.8h,  v1.h[3]		// i[3]k[7-0]
	fmla	v28.8h, v5.8h,  v1.h[0]		// i[0]k[f-8]
	fmla	v29.8h, v5.8h,  v1.h[1]		// i[1]k[f-8]
	fmla	v30.8h, v5.8h,  v1.h[2]		// i[2]k[f-8]
	fmla	v31.8h, v5.8h,  v1.h[3]		// i[3]k[f-8]

	prfm	pldl1keep, [x2, 0x280]
	fmla	v24.8h, v6.8h,  v1.h[4]		// i[0]k[7-0]
	fmla	v25.8h, v6.8h,  v1.h[5]		// i[1]k[7-0]
	prfm	pldl1keep, [x2, 0x2c0]
	fmla	v26.8h, v6.8h,  v1.h[6]		// i[2]k[7-0]
	fmla	v27.8h, v6.8h,  v1.h[7]		// i[3]k[7-0]
	fmla	v28.8h, v7.8h,  v1.h[4]		// i[0]k[f-8]
	fmla	v29.8h, v7.8h,  v1.h[5]		// i[1]k[f-8]
	fmla	v30.8h, v7.8h,  v1.h[6]		// i[2]k[f-8]
	fmla	v31.8h, v7.8h,  v1.h[7]		// i[3]k[f-8]
	b.ne	loop4


loop4_end:
	cbz	x10, fused_relu

loop1:
	ldr	d0, [x1], 0x8			// d0=i[3-0]
	ldp	q4, q5, [x2], 0x20		// q4=k[7-0] q5=k[f-8] 
        subs    x10, x10 ,0x1
	fmla	v24.8h, v4.8h,  v0.h[0]		// i[0]k[7-0]
	fmla	v25.8h, v4.8h,  v0.h[1]		// i[1]k[7-0]
	fmla	v26.8h, v4.8h,  v0.h[2]		// i[2]k[7-0]
	fmla	v27.8h, v4.8h,  v0.h[3]		// i[3]k[7-0]
	fmla	v28.8h, v5.8h,  v0.h[0]		// i[0]k[f-8]
	fmla	v29.8h, v5.8h,  v0.h[1]		// i[1]k[f-8]
	fmla	v30.8h, v5.8h,  v0.h[2]		// i[2]k[f-8]
	fmla	v31.8h, v5.8h,  v0.h[3]		// i[3]k[f-8]

        b.ne    loop1
	

fused_relu:
	cmp     x6, 0
    blt     save_result

    scvtf   h1, x6
	movi	d0, 0
	fmax	v24.8h, v24.8h, v0.8h
	fmax	v25.8h, v25.8h, v0.8h
	fmax	v26.8h, v26.8h, v0.8h
	fmax	v27.8h, v27.8h, v0.8h
	fmax	v28.8h, v28.8h, v0.8h
	fmax	v29.8h, v29.8h, v0.8h
	fmax	v30.8h, v30.8h, v0.8h
	fmax	v31.8h, v31.8h, v0.8h
	
    beq     save_result
    dup     v1.8h, v1.h[0]
    fmin	v24.8h, v24.8h, v1.8h
    fmin	v25.8h, v25.8h, v1.8h
    fmin	v26.8h, v26.8h, v1.8h
    fmin	v27.8h, v27.8h, v1.8h
    fmin	v28.8h, v28.8h, v1.8h
    fmin	v29.8h, v29.8h, v1.8h
    fmin	v30.8h, v30.8h, v1.8h
    fmin	v31.8h, v31.8h, v1.8h


save_result:
	// store result
	// x4 x11 x12 x13 , x15 x16 x17 x18, x9 x10 x0 x1, x4 x11 x12 x13 as base address

        st4     {v24.h,v25.h,v26.h,v27.h}[0], [x4]
        st4     {v24.h,v25.h,v26.h,v27.h}[1], [x11]
        st4     {v24.h,v25.h,v26.h,v27.h}[2], [x12]
        st4     {v24.h,v25.h,v26.h,v27.h}[3], [x13]
	add	x18,x13, x5, LSL 2		// x18 = output + ouput_xy * 7

        st4     {v24.h,v25.h,v26.h,v27.h}[4], [x15]
	add	x9,  x4, x5, LSL 3		// x9 = output + ouput_xy * 8
        st4     {v24.h,v25.h,v26.h,v27.h}[5], [x16]
	add	x10,x11, x5, LSL 3
        st4     {v24.h,v25.h,v26.h,v27.h}[6], [x17]
	add	x0, x12, x5, LSL 3
        st4     {v24.h,v25.h,v26.h,v27.h}[7], [x18]
	add	x1, x13, x5, LSL 3

        st4     {v28.h,v29.h,v30.h,v31.h}[0], [x9]
	add	x4, x15, x5, LSL 3
        st4     {v28.h,v29.h,v30.h,v31.h}[1], [x10]
	add	x11,x16, x5, LSL 3
        st4     {v28.h,v29.h,v30.h,v31.h}[2], [x0]
	add	x12,x17, x5, LSL 3
        st4     {v28.h,v29.h,v30.h,v31.h}[3], [x1]
	add	x13,x18, x5, LSL 3

        st4     {v28.h,v29.h,v30.h,v31.h}[4], [x4]
        st4     {v28.h,v29.h,v30.h,v31.h}[5], [x11]
        st4     {v28.h,v29.h,v30.h,v31.h}[6], [x12]
        st4     {v28.h,v29.h,v30.h,v31.h}[7], [x13]

	ret

        .end

