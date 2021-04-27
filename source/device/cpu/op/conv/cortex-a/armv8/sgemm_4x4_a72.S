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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xiaowei@openailab.com
 */
//
// 4*4 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                   --
//    | i0 - - - - - - |      |  k0  k1  k2  k3 |     |  b0  b1  b2  b3 |         | i0k0 i0k1 i0k2 i0k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i1k0 i1k1 i1k2 i1k3 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                     |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i2k0 i2k1 i2k2 i2k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                     |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i3k0 i3k1 i3k2 i3k3 |
//    --              --      --               --     --               --         --                   --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 4         p = kernel size
//
//
// optimised for Cortex-A72 pipeline 18 cycle per loop (4*4*4 dot product)
//
// input:  
//         x0 arg0  biases address {b0,b1,b2,b3}  nullptr means no biases 
//         x1 arg1  input  address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x2 arg2  kernel address {k[0-3][0],k[0-3][1],k[0-3][2],k[0-3][3],...}
//         x3 arg3  kernel size
//         x4 arg4  output address 
//                  indirect save: output {i[0-3]k[0],i[0-3]k[1],i[0-3]k[2],i[0-3]k[3]}
//                    direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                 output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                 output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                 output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         x5 arg5  output xy
//         x6 arg6  activation flag     relu layers is integrated after convolution
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
// x6        fused relu flag
// x9 ~ x10  temp loop counter
// x11~ x13  temp output save address
// x7~8 14~15 ot used

//
// v0-3 4S data of input0   {i3   i2   i1   i0}
// v4-7 4S kernal data      {k3   k2   k1   k0}
// v8~v15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20~V31 not used

        .section .text,"ax"
        .align 5

        .type sgemm_4x4_a72 STT_FUNC
        .global sgemm_4x4_a72
        .hidden sgemm_4x4_a72
sgemm_4x4_a72:
        // bring some code ahead to reduce dependency
	prfm	pldl1keep, [x1, 0x40]
	lsl     x5, x5, 2                       // x5  = output_xy
	prfm	pldl1keep, [x2, 0x40]
	cmp	x3, 0x4
        // initial biases
	cbz	x0,  non_biases
        ld4r    {v16.4s,v17.4s,v18.4s,v19.4s}, [x0]
	b	convoluation_start
	
non_biases:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0

convoluation_start:
        add     x11,x4, x5                      // x11 = output + ouput_xy
	and	x10,x3, 0x3
	b.lt	loop4_end
	lsr	x9 ,x3, 0x2

// main loop     each loop generate dot prodcut for 4x4SFP
loop4:  
	subs	x9 ,x9 ,0x1

	ldr	q0, [x1]			// q0=i[3-0]
	ldp	q4, q5, [x2]			// q4=k[3-0] 
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	ldr	q1, [x1, 0x10]			// q1=i[3-0]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]

	ldp	q2, q3, [x1, 0x20]		// q2=i[3-0] q3=i[3-0]
	fmla	v16.4s, v1.4s,  v5.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v5.s[1]		// i[3-0]k[1]
	ldp	q6, q7, [x2, 0x20]		// q6=k[3-0] q7=q7=k[3-0]
	fmla	v18.4s, v1.4s,  v5.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v5.s[3]		// i[3-0]k[3]

	fmla	v16.4s, v2.4s,  v6.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v2.4s,  v6.s[1]		// i[3-0]k[1]
	prfm	pldl1keep, [x1, 0x140]
	add	x1, x1, 0x40
	fmla	v18.4s, v2.4s,  v6.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v2.4s,  v6.s[3]		// i[3-0]k[3]

	prfm	pldl1keep, [x2, 0x140]
	add	x2, x2, 0x40
	fmla	v16.4s, v3.4s,  v7.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v3.4s,  v7.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v3.4s,  v7.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v3.4s,  v7.s[3]		// i[3-0]k[3]

	b.ne	loop4


loop4_end:
        add     x12,x4, x5, LSL 1               // x12 = output + ouput_xy * 2
	cbz	x10, activation

loop1:
	subs	x10 ,x10 ,0x1
	ldr     q0, [x1], 0x10                  // q0=i[3-0]
        ldr     q4, [x2], 0x10                  // q4=k[3-0]
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[0]k[3-0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[1]k[3-0]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[2]k[3-0]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3]k[3-0]
	b.ne	loop1

activation:
        add     x13,x11,x5, LSL 1               // x13 = output + ouput_xy * 2
		cmp     w6,0
        blt     save_result

        movi    d0, 0
		scvtf   s1, w6

        fmax    v16.4s, v16.4s, v0.4s
        fmax    v17.4s, v17.4s, v0.4s
        fmax    v18.4s, v18.4s, v0.4s
        fmax    v19.4s, v19.4s, v0.4s

		beq     save_result

		dup     v1.4s,v1.s[0]

        fmin    v16.4s, v16.4s, v1.4s
        fmin    v17.4s, v17.4s, v1.4s
        fmin    v18.4s, v18.4s, v1.4s
        fmin    v19.4s, v19.4s, v1.4s

save_result:
	// store result
    // x4 x11 x12 13 as base address
	cmp     w7, 0
    beq     save_result_nchw
    st4     {v16.s,v17.s,v18.s,v19.s}[0], [x4]
    st4     {v16.s,v17.s,v18.s,v19.s}[1], [x11]
    st4     {v16.s,v17.s,v18.s,v19.s}[2], [x12]
    st4     {v16.s,v17.s,v18.s,v19.s}[3], [x13]

    b       end

save_result_nchw:
        str     q16, [x4]
        str     q17, [x11]
        str     q18, [x12]
        str     q19, [x13]

end:
	ret
    .end

