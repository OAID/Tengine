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
 * Author: haoluo@openailab.com
 */

//
// depthwise convolution kernel size 5x5 stride 1
// input:
//        x0     arg0  input data address
//        x1     arg1  kernel data address
//        x2     arg3  bias
//        x3     arg2  output data address
//        x4     arg4  output h
//        x5     arg5  output w
//        x6     arg6  activation
// output: no
//
// register definition
//        x0     intput data address for every channel
//        x1     kernel pointer
//        x2     bias
//        x3     output data address for every channel
//        x4     output h
//        x5     output w
//        x6     activation
//        x7     input_1
//        x9    input_2
//        x10    input_3
//        x11    input_4
//        x12    input_5
//        x13    output_h & 0x3
//        x14    output_w & 0x3
//        x15    output_w*4
//
// kernel q0 0-3
//        q1 4-7
//        q2 8-11
//        q3 12-15
//        q4 16-19
//        q5 20-23
//        q6 24-27
//
// input  q8 ~ q19
//
// bias   q29
// relu   q30, q31
//
// output q24,q25
//
//


#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k5s1
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

	movi	d30, 0
	scvtf   s31, w6
	dup     v31.4s, v31.s[0]

	movi	d29, 0
	cbz	    x2, no_biases	
	ld1r	{v29.4s}, [x2], 0x4

no_biases:
    ldp     q0, q1, [x1]
    ldp     q2, q3, [x1, 0x20]
    ldp     q4, q5, [x1, 0x40]
    ldr     q6, [x1, 0x60]

    add     x15, x5, 0x4
    cmp     x4, 0x2
    lsl     x15, x15, 0x2
    blt     loop_h_1

loop_h_2:
    add     x7, x0, x15
    add     x9, x0, x15, lsl 0x1
    add     x10, x7, x15, lsl 0x1
    add     x11, x0, x15, lsl 0x2
    add     x12, x7, x15, lsl 0x2
    add     x2, x3, x5, lsl 0x2

    ldr     q8, [x0]
    ldr     q10, [x7]
    ldr     q12, [x9]
    ldr     q14, [x10]
    ldr     q16, [x11]
    ldr     q18, [x12]
    lsr     x14, x5, 0x2
    cmp     x14, 0
    beq     loop_w_1

loop_w_4:
    ldr     q9, [x0, 0x10]
    ldr     q11, [x7, 0x10]
    ldr     q13, [x9, 0x10]
    ldr     q15, [x10, 0x10]
    ldr     q17, [x11, 0x10]
    ldr     q19, [x12, 0x10]
    ext     v20.16b, v8.16b, v9.16b, 4
    fmul    v24.4s, v8.4s, v0.s[0]
    ext     v21.16b, v8.16b, v9.16b, 8
    fmla    v24.4s, v20.4s,  v0.s[1]
    ext     v22.16b, v8.16b, v9.16b, 12
    fmla    v24.4s, v21.4s,  v0.s[2]
    mov     v8.16b, v9.16b
    fmla    v24.4s, v22.4s,  v0.s[3]
    add     x0, x0, 0x10
    fmla    v24.4s, v9.4s,  v1.s[0]

    add     x7, x7, 0x10
    ext     v20.16b, v10.16b, v11.16b, 4
    ext     v21.16b, v10.16b, v11.16b, 8
    ext     v22.16b, v10.16b, v11.16b, 12

    fmla    v24.4s, v10.4s, v1.s[1]
    fmul    v25.4s, v10.4s, v0.s[0]
    fmla    v24.4s, v20.4s, v1.s[2]
    fmla    v25.4s, v20.4s,  v0.s[1]
    fmla    v24.4s, v21.4s, v1.s[3]
    fmla    v25.4s, v21.4s,  v0.s[2]
    fmla    v24.4s, v22.4s, v2.s[0]
    fmla    v25.4s, v22.4s,  v0.s[3]
    mov     v10.16b, v11.16b
    fmla    v24.4s, v11.4s, v2.s[1]
    fmla    v25.4s, v11.4s,  v1.s[0]

    add     x9, x9, 0x10
    ext     v20.16b, v12.16b, v13.16b, 4
    ext     v21.16b, v12.16b, v13.16b, 8
    ext     v22.16b, v12.16b, v13.16b, 12

    fmla    v24.4s, v12.4s,  v2.s[2]
    fmla    v25.4s, v12.4s, v1.s[1]
    fmla    v24.4s, v20.4s,  v2.s[3]
    fmla    v25.4s, v20.4s, v1.s[2]
    fmla    v24.4s, v21.4s,  v3.s[0]
    fmla    v25.4s, v21.4s, v1.s[3]
    fmla    v24.4s, v22.4s,  v3.s[1]
    fmla    v25.4s, v22.4s, v2.s[0]
    fmla    v24.4s, v13.4s,  v3.s[2]
    fmla    v25.4s, v13.4s, v2.s[1]
    mov     v12.16b, v13.16b

    add     x10, x10, 0x10
    ext     v20.16b, v14.16b, v15.16b, 4
    ext     v21.16b, v14.16b, v15.16b, 8
    ext     v22.16b, v14.16b, v15.16b, 12

    fmla    v24.4s, v14.4s,  v3.s[3]
    fmla    v25.4s, v14.4s,  v2.s[2]
    fmla    v24.4s, v20.4s,  v4.s[0]
    fmla    v25.4s, v20.4s,  v2.s[3]
    fmla    v24.4s, v21.4s,  v4.s[1]
    fmla    v25.4s, v21.4s,  v3.s[0]
    fmla    v24.4s, v22.4s,  v4.s[2]
    fmla    v25.4s, v22.4s,  v3.s[1]
    mov     v14.16b, v15.16b
    fmla    v24.4s, v15.4s,  v4.s[3]
    fmla    v25.4s, v15.4s,  v3.s[2]

    add     x11, x11, 0x10
    ext     v20.16b, v16.16b, v17.16b, 4
    ext     v21.16b, v16.16b, v17.16b, 8
    ext     v22.16b, v16.16b, v17.16b, 12
    fmla    v24.4s, v16.4s,  v5.s[0]
    fmla    v25.4s, v16.4s,  v3.s[3]
    fmla    v24.4s, v20.4s,  v5.s[1]
    fmla    v25.4s, v20.4s,  v4.s[0]
    fmla    v24.4s, v21.4s,  v5.s[2]
    fmla    v25.4s, v21.4s,  v4.s[1]
    fmla    v24.4s, v22.4s,  v5.s[3]
    fmla    v25.4s, v22.4s,  v4.s[2]
    fmla    v24.4s, v17.4s,  v6.s[0]
    fmla    v25.4s, v17.4s,  v4.s[3]

    mov     v16.16b, v17.16b

    ext     v20.16b, v18.16b, v19.16b, 4
    fmla    v25.4s, v18.4s,  v5.s[0]
    ext     v21.16b, v18.16b, v19.16b, 8
    fmla    v25.4s, v20.4s,  v5.s[1]
    ext     v22.16b, v18.16b, v19.16b, 12
    fmla    v25.4s, v21.4s,  v5.s[2]
    add     x12, x12, 0x10
    fmla    v25.4s, v22.4s,  v5.s[3]
    mov     v18.16b, v19.16b
    fmla    v25.4s, v19.4s,  v6.s[0]

	fadd	v24.4s, v24.4s, v29.4s
	fadd	v25.4s, v25.4s, v29.4s

	cmp     w6,0
    blt     save_result_4
    fmax    v24.4s, v24.4s, v30.4s
    fmax    v25.4s, v25.4s, v30.4s
    beq     save_result_4
    fmin	v24.4s, v24.4s, v31.4s
    fmin	v25.4s, v25.4s, v31.4s

save_result_4:
    str     q24, [x3]
    str     q25, [x2]
    add     x3, x3, 0x10
    add     x2, x2, 0x10

    subs    x14, x14, 0x1
    bne     loop_w_4
loop_w_1:
    and     x14, x5, 0x3
    cmp     x14, 0
    beq     loop_h_2_end

    ldr     q9, [x0, 0x10]
    ldr     q11, [x7, 0x10]
    ext     v20.16b, v8.16b, v9.16b, 4
    fmul    v24.4s, v8.4s, v0.s[0]
    ext     v21.16b, v8.16b, v9.16b, 8
    fmla    v24.4s, v20.4s,  v0.s[1]
    ext     v22.16b, v8.16b, v9.16b, 12
    fmla    v24.4s, v21.4s,  v0.s[2]
    fmla    v24.4s, v22.4s,  v0.s[3]
    fmla    v24.4s, v9.4s,  v1.s[0]
    ldr     q13, [x9, 0x10]

    ext     v20.16b, v10.16b, v11.16b, 4
    ext     v21.16b, v10.16b, v11.16b, 8
    ext     v22.16b, v10.16b, v11.16b, 12

    fmla    v24.4s, v10.4s, v1.s[1]
    fmul    v25.4s, v10.4s, v0.s[0]
    fmla    v24.4s, v20.4s, v1.s[2]
    fmla    v25.4s, v20.4s,  v0.s[1]
    fmla    v24.4s, v21.4s, v1.s[3]
    fmla    v25.4s, v21.4s,  v0.s[2]
    fmla    v24.4s, v22.4s, v2.s[0]
    fmla    v25.4s, v22.4s,  v0.s[3]
    fmla    v24.4s, v11.4s, v2.s[1]
    fmla    v25.4s, v11.4s,  v1.s[0]

    ldr     q15, [x10, 0x10]
    ext     v20.16b, v12.16b, v13.16b, 4
    ext     v21.16b, v12.16b, v13.16b, 8
    ext     v22.16b, v12.16b, v13.16b, 12

    fmla    v24.4s, v12.4s,  v2.s[2]
    fmla    v25.4s, v12.4s, v1.s[1]
    fmla    v24.4s, v20.4s,  v2.s[3]
    fmla    v25.4s, v20.4s, v1.s[2]
    fmla    v24.4s, v21.4s,  v3.s[0]
    fmla    v25.4s, v21.4s, v1.s[3]
    fmla    v24.4s, v22.4s,  v3.s[1]
    fmla    v25.4s, v22.4s, v2.s[0]
    fmla    v24.4s, v13.4s,  v3.s[2]
    fmla    v25.4s, v13.4s, v2.s[1]

    ldr     q17, [x11, 0x10]
    ext     v20.16b, v14.16b, v15.16b, 4
    ext     v21.16b, v14.16b, v15.16b, 8
    ext     v22.16b, v14.16b, v15.16b, 12

    fmla    v24.4s, v14.4s,  v3.s[3]
    fmla    v25.4s, v14.4s,  v2.s[2]
    fmla    v24.4s, v20.4s,  v4.s[0]
    fmla    v25.4s, v20.4s,  v2.s[3]
    fmla    v24.4s, v21.4s,  v4.s[1]
    fmla    v25.4s, v21.4s,  v3.s[0]
    fmla    v24.4s, v22.4s,  v4.s[2]
    fmla    v25.4s, v22.4s,  v3.s[1]
    fmla    v24.4s, v15.4s,  v4.s[3]
    fmla    v25.4s, v15.4s,  v3.s[2]

    ldr     q19, [x12, 0x10]
    ext     v20.16b, v16.16b, v17.16b, 4
    ext     v21.16b, v16.16b, v17.16b, 8
    ext     v22.16b, v16.16b, v17.16b, 12
    fmla    v24.4s, v16.4s,  v5.s[0]
    fmla    v25.4s, v16.4s,  v3.s[3]
    fmla    v24.4s, v20.4s,  v5.s[1]
    fmla    v25.4s, v20.4s,  v4.s[0]
    fmla    v24.4s, v21.4s,  v5.s[2]
    fmla    v25.4s, v21.4s,  v4.s[1]
    fmla    v24.4s, v22.4s,  v5.s[3]
    fmla    v25.4s, v22.4s,  v4.s[2]
    fmla    v24.4s, v17.4s,  v6.s[0]
    fmla    v25.4s, v17.4s,  v4.s[3]


    ext     v20.16b, v18.16b, v19.16b, 4
    fmla    v25.4s, v18.4s,  v5.s[0]
    ext     v21.16b, v18.16b, v19.16b, 8
    fmla    v25.4s, v20.4s,  v5.s[1]
    ext     v22.16b, v18.16b, v19.16b, 12
    fmla    v25.4s, v21.4s,  v5.s[2]
    add     x0, x0, x14, lsl 0x2
    fmla    v25.4s, v22.4s,  v5.s[3]
	fadd	v24.4s, v24.4s, v29.4s
    fmla    v25.4s, v19.4s,  v6.s[0]
	cmp     w6,0
	fadd	v25.4s, v25.4s, v29.4s

    blt     save_result_1
    fmax    v24.4s, v24.4s, v30.4s
    fmax    v25.4s, v25.4s, v30.4s
    beq     save_result_4
    fmin	v24.4s, v24.4s, v31.4s
    fmin	v25.4s, v25.4s, v31.4s
    
save_result_1:
    st1     {v24.s}[0], [x3]
    st1     {v25.s}[0], [x2]
    add     x3, x3, 0x4
    add     x2, x2, 0x4
    ext     v24.16b, v24.16b, v24.16b, 4
    ext     v25.16b, v25.16b, v25.16b, 4
    subs    x14, x14, 0x1
    bne     save_result_1

loop_h_2_end:
    add     x0, x0, 0x10
    add     x0, x0, x15
    mov     x3, x2

    sub     x4, x4, 2
    cmp     x4, 1
    bgt     loop_h_2

loop_h_1:
    cmp     x4, 0
    beq     loop_h_1_end
    add     x7, x0, x15
    add     x9, x0, x15, lsl 0x1
    add     x10, x7, x15, lsl 0x1
    add     x11, x0, x15, lsl 0x2
    ldr     q8, [x0]
    ldr     q10, [x7]
    ldr     q12, [x9]
    ldr     q14, [x10]
    ldr     q16, [x11]

    lsr     x14, x5, 0x2
    cmp     x14, 0
    beq     loop_h1_w1
    
loop_h1_w_4:
    ldr     q9, [x0, 0x10]
    ldr     q11, [x7, 0x10]
    ldr     q13, [x9, 0x10]
    ldr     q15, [x10, 0x10]
    ldr     q17, [x11, 0x10]
    ext     v20.16b, v8.16b, v9.16b, 4
    fmul    v24.4s, v8.4s, v0.s[0]
    ext     v21.16b, v8.16b, v9.16b, 8
    fmla    v24.4s, v20.4s,  v0.s[1]
    ext     v22.16b, v8.16b, v9.16b, 12
    fmla    v24.4s, v21.4s,  v0.s[2]
    mov     v8.16b, v9.16b
    fmla    v24.4s, v22.4s,  v0.s[3]
    add     x0, x0, 0x10
    fmla    v24.4s, v9.4s,  v1.s[0]

    ext     v20.16b, v10.16b, v11.16b, 4
    fmla    v24.4s, v10.4s, v1.s[1]
    ext     v21.16b, v10.16b, v11.16b, 8
    fmla    v24.4s, v20.4s, v1.s[2]
    ext     v22.16b, v10.16b, v11.16b, 12
    fmla    v24.4s, v21.4s, v1.s[3]
    add     x7, x7, 0x10
    fmla    v24.4s, v22.4s, v2.s[0]
    mov     v10.16b, v11.16b
    fmla    v24.4s, v11.4s, v2.s[1]

    ext     v20.16b, v12.16b, v13.16b, 4
    fmla    v24.4s, v12.4s,  v2.s[2]
    ext     v21.16b, v12.16b, v13.16b, 8
    fmla    v24.4s, v20.4s,  v2.s[3]
    ext     v22.16b, v12.16b, v13.16b, 12
    fmla    v24.4s, v21.4s,  v3.s[0]
    add     x9, x9, 0x10
    fmla    v24.4s, v22.4s,  v3.s[1]
    mov     v12.16b, v13.16b
    fmla    v24.4s, v13.4s,  v3.s[2]

    ext     v20.16b, v14.16b, v15.16b, 4
    fmla    v24.4s, v14.4s,  v3.s[3]
    ext     v21.16b, v14.16b, v15.16b, 8
    fmla    v24.4s, v20.4s,  v4.s[0]
    ext     v22.16b, v14.16b, v15.16b, 12
    fmla    v24.4s, v21.4s,  v4.s[1]
    add     x10, x10, 0x10
    fmla    v24.4s, v22.4s,  v4.s[2]
    mov     v14.16b, v15.16b
    fmla    v24.4s, v15.4s,  v4.s[3]

    ext     v20.16b, v16.16b, v17.16b, 4
    fmla    v24.4s, v16.4s,  v5.s[0]
    ext     v21.16b, v16.16b, v17.16b, 8
    fmla    v24.4s, v20.4s,  v5.s[1]
    ext     v22.16b, v16.16b, v17.16b, 12
    fmla    v24.4s, v21.4s,  v5.s[2]
    add     x11, x11, 0x10
    fmla    v24.4s, v22.4s,  v5.s[3]
    mov     v16.16b, v17.16b
    fmla    v24.4s, v17.4s,  v6.s[0]

	fadd	v24.4s, v24.4s, v29.4s

	cmp     w6,0
    blt     save_result_h1_4
    fmax    v24.4s, v24.4s, v30.4s
    beq     save_result_h1_4
    fmin	v24.4s, v24.4s, v31.4s

save_result_h1_4:
    str     q24, [x3]
    add     x3, x3, 0x10

    subs    x14, x14, 0x1
    bne     loop_h1_w_4
loop_h1_w1:
    and     x14, x5, 0x3
    cmp     x14, 0
    beq     loop_h_1_end

    ldr     q9, [x0, 0x10]
    ldr     q11, [x7, 0x10]
    ldr     q13, [x9, 0x10]
    ldr     q15, [x10, 0x10]
    ldr     q17, [x11, 0x10]
    ext     v20.16b, v8.16b, v9.16b, 4
    fmul    v24.4s, v8.4s, v0.s[0]
    ext     v21.16b, v8.16b, v9.16b, 8
    fmla    v24.4s, v20.4s,  v0.s[1]
    ext     v22.16b, v8.16b, v9.16b, 12
    fmla    v24.4s, v21.4s,  v0.s[2]
    fmla    v24.4s, v22.4s,  v0.s[3]
    fmla    v24.4s, v9.4s,  v1.s[0]

    ext     v20.16b, v10.16b, v11.16b, 4
    fmla    v24.4s, v10.4s, v1.s[1]
    ext     v21.16b, v10.16b, v11.16b, 8
    fmla    v24.4s, v20.4s, v1.s[2]
    ext     v22.16b, v10.16b, v11.16b, 12
    fmla    v24.4s, v21.4s, v1.s[3]
    fmla    v24.4s, v22.4s, v2.s[0]
    fmla    v24.4s, v11.4s, v2.s[1]

    ext     v20.16b, v12.16b, v13.16b, 4
    fmla    v24.4s, v12.4s,  v2.s[2]
    ext     v21.16b, v12.16b, v13.16b, 8
    fmla    v24.4s, v20.4s,  v2.s[3]
    ext     v22.16b, v12.16b, v13.16b, 12
    fmla    v24.4s, v21.4s,  v3.s[0]
    fmla    v24.4s, v22.4s,  v3.s[1]
    fmla    v24.4s, v13.4s,  v3.s[2]

    ext     v20.16b, v14.16b, v15.16b, 4
    fmla    v24.4s, v14.4s,  v3.s[3]
    ext     v21.16b, v14.16b, v15.16b, 8
    fmla    v24.4s, v20.4s,  v4.s[0]
    ext     v22.16b, v14.16b, v15.16b, 12
    fmla    v24.4s, v21.4s,  v4.s[1]
    fmla    v24.4s, v22.4s,  v4.s[2]
    fmla    v24.4s, v15.4s,  v4.s[3]

    ext     v20.16b, v16.16b, v17.16b, 4
    fmla    v24.4s, v16.4s,  v5.s[0]
    ext     v21.16b, v16.16b, v17.16b, 8
    fmla    v24.4s, v20.4s,  v5.s[1]
    ext     v22.16b, v16.16b, v17.16b, 12
    fmla    v24.4s, v21.4s,  v5.s[2]
    fmla    v24.4s, v22.4s,  v5.s[3]
    fmla    v24.4s, v17.4s,  v6.s[0]

	fadd	v24.4s, v24.4s, v29.4s

	cmp     w6,0
    blt     save_result_h1_1
    fmax    v24.4s, v24.4s, v30.4s
    beq     save_result_h1_1
    fmin	v24.4s, v24.4s, v31.4s

save_result_h1_1:
    st1     {v24.s}[0], [x3]
    add     x3, x3, 0x4
    ext     v24.16b, v24.16b, v24.16b, 4
    subs    x14, x14, 0x1
    bne     save_result_h1_1

loop_h_1_end:

	ldp	d8, d9, [sp]
	ldp	d10,d11,[sp, 0x10]
	ldp	d12,d13,[sp, 0x20]
	ldp	d14,d15,[sp, 0x30]
	add	sp, sp, 0x40

	ret
