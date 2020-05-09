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
 * Copyright (c) 2019, Open AI Lab
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/

//
// 4*16 half precise floating point matric multiplication
//
//    --              --      --               --       --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  kf |       | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |       |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |       | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |   =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |       | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |       |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |       | i3k0 i3k1 .. i3kf |
//    --              --      --               --       --                 --
//      input 4 x p             kernel p x 16                output 4 x 16           p = cin
//
//
// optimised for a76 pipeline 16 cycle per loop (4*16*4 dot product)
//
// input: 
//         x0 arg0  output address : continue save
//         x1 arg1  input  address {i[0-3][0],i1[0-3][1],i[0-3][2],i[0-3][3],i[0-3][4],...}
//         x2 arg2  kernel address {k[0-15][0],k[0-15][1],k[0-15][2],k[0-15][3],...}
//         x3 arg3  in_channel (cin)

//
// register definition
// x0        output start address
// x1        input start address
// x2        kernel start address
// x3        in_channel (cin)

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

    .type wino_hgemm_4x16_fp16 STT_FUNC
    .global wino_hgemm_4x16_fp16

wino_hgemm_4x16_fp16:

    cmp    x3, 0x4
	movi	d24, 0x0
    add x11,x0,#0x120
	movi	d25, 0x0
    add x12,x0,#0x240
	movi	d26, 0x0
    add x13,x0,#0x360
	movi	d27, 0x0
    mov x18,#0x480
	movi	d28, 0x0
    add x14,x0,x18
	movi	d29, 0x0
    add x15,x14,#0x120
	movi	d30, 0x0
    add x16,x14,#0x240
	movi	d31, 0x0
    add x17,x14,#0x360

    and    x10,x3, 0x3
    b.lt    loop4_end
    lsr    x9, x3, 0x2

// main loop     each loop generate dot prodcut for 4x16x4SP
loop4:  
    ldnp    q0, q1, [x1]              // q0=i[3-0][1-0] q1=i[3-0][3-2]
    ldp    q4, q5, [x2]                    // q4=k[7-0][0] q5=k[f-8][0] 
    ldp    q6, q7, [x2, 0x20]              // q6=k[7-0][1] q7=k[f-8][1] 
    subs    x9, x9, 0x1
    fmla    v24.8h, v4.8h,  v0.h[0]        // i[0]k[7-0]
    fmla    v25.8h, v4.8h,  v0.h[1]        // i[1]k[7-0]
    fmla    v26.8h, v4.8h,  v0.h[2]        // i[2]k[7-0]
    fmla    v27.8h, v4.8h,  v0.h[3]        // i[3]k[7-0]
    fmla    v28.8h, v5.8h,  v0.h[0]        // i[0]k[f-8]
    fmla    v29.8h, v5.8h,  v0.h[1]        // i[1]k[f-8]
    add	x1, x1, 0x20
    fmla    v30.8h, v5.8h,  v0.h[2]        // i[2]k[f-8]
    fmla    v31.8h, v5.8h,  v0.h[3]        // i[3]k[f-8]

    fmla    v24.8h, v6.8h,  v0.h[4]        // i[0]k[7-0]
    fmla    v25.8h, v6.8h,  v0.h[5]        // i[1]k[7-0]
    fmla    v26.8h, v6.8h,  v0.h[6]        // i[2]k[7-0]
    fmla    v27.8h, v6.8h,  v0.h[7]        // i[3]k[7-0]
    fmla    v28.8h, v7.8h,  v0.h[4]        // i[0]k[f-8]
    fmla    v29.8h, v7.8h,  v0.h[5]        // i[1]k[f-8]
    fmla    v30.8h, v7.8h,  v0.h[6]        // i[2]k[f-8]
    fmla    v31.8h, v7.8h,  v0.h[7]        // i[3]k[f-8]

    ldp    q4, q5, [x2, 0x40]              // q4=k[7-0][2] q5=k[f-8][2]
    ldp    q6, q7, [x2, 0x60]              // q6=k[7-0][3] q7=k[f-8][3]
    fmla    v24.8h, v4.8h,  v1.h[0]        // i[0]k[7-0]
    fmla    v25.8h, v4.8h,  v1.h[1]        // i[1]k[7-0]
    add    x2, x2, 0x80
    fmla    v26.8h, v4.8h,  v1.h[2]        // i[2]k[7-0]
    fmla    v27.8h, v4.8h,  v1.h[3]        // i[3]k[7-0]
    fmla    v28.8h, v5.8h,  v1.h[0]        // i[0]k[f-8]
    fmla    v29.8h, v5.8h,  v1.h[1]        // i[1]k[f-8]
    fmla    v30.8h, v5.8h,  v1.h[2]        // i[2]k[f-8]
    fmla    v31.8h, v5.8h,  v1.h[3]        // i[3]k[f-8]

    prfm	pldl1keep, [x2, 0x280]
    fmla    v24.8h, v6.8h,  v1.h[4]        // i[0]k[7-0]
    fmla    v25.8h, v6.8h,  v1.h[5]        // i[1]k[7-0]
    prfm	pldl1keep, [x2, 0x2c0]
    fmla    v26.8h, v6.8h,  v1.h[6]        // i[2]k[7-0]
    fmla    v27.8h, v6.8h,  v1.h[7]        // i[3]k[7-0]
    fmla    v28.8h, v7.8h,  v1.h[4]        // i[0]k[f-8]
    fmla    v29.8h, v7.8h,  v1.h[5]        // i[1]k[f-8]
    fmla    v30.8h, v7.8h,  v1.h[6]        // i[2]k[f-8]
    fmla    v31.8h, v7.8h,  v1.h[7]        // i[3]k[f-8]
    b.ne    loop4


loop4_end:
    cbz    x10, save_result

loop1:
    ldr    d0, [x1], 0x8                   // d0=i[3-0]
    ldp    q4, q5, [x2], 0x20              // q4=k[7-0] q5=k[f-8] 
    subs    x10, x10 ,0x1
    fmla    v24.8h, v4.8h,  v0.h[0]        // i[0]k[7-0]
    fmla    v25.8h, v4.8h,  v0.h[1]        // i[1]k[7-0]
    fmla    v26.8h, v4.8h,  v0.h[2]        // i[2]k[7-0]
    fmla    v27.8h, v4.8h,  v0.h[3]        // i[3]k[7-0]
    fmla    v28.8h, v5.8h,  v0.h[0]        // i[0]k[f-8]
    fmla    v29.8h, v5.8h,  v0.h[1]        // i[1]k[f-8]
    fmla    v30.8h, v5.8h,  v0.h[2]        // i[2]k[f-8]
    fmla    v31.8h, v5.8h,  v0.h[3]        // i[3]k[f-8]
    b.ne    loop1
	
save_result:
    cmp     w4,0
    beq     direct_save

    stride_save:
       
    st4     {v24.h,v25.h,v26.h,v27.h}[0], [x0]
    st4     {v24.h,v25.h,v26.h,v27.h}[1], [x11]   //each line 36*4 data
    st4     {v24.h,v25.h,v26.h,v27.h}[2], [x12]
    st4     {v24.h,v25.h,v26.h,v27.h}[3], [x13]
    add x9,x14,x18
    st4     {v24.h,v25.h,v26.h,v27.h}[4], [x14]
    add x11,x9,#0x120
    st4     {v24.h,v25.h,v26.h,v27.h}[5], [x15]   //each line 36*4 data
    add x12,x9,#0x240
    st4     {v24.h,v25.h,v26.h,v27.h}[6], [x16]
    add x13,x9,#0x360
    st4     {v24.h,v25.h,v26.h,v27.h}[7], [x17]
  
    add x10,x9,x18
    st4     {v28.h,v29.h,v30.h,v31.h}[0], [x9]
    add x14,x10,#0x120
    st4     {v28.h,v29.h,v30.h,v31.h}[1], [x11]   //each line 36*4 data
    add x15,x10,#0x240
    st4     {v28.h,v29.h,v30.h,v31.h}[2], [x12]
    add x16,x10,#0x360
    st4     {v28.h,v29.h,v30.h,v31.h}[3], [x13]

    st4     {v28.h,v29.h,v30.h,v31.h}[4], [x10]
    st4     {v28.h,v29.h,v30.h,v31.h}[5], [x14]   //each line 36*4 data
    st4     {v28.h,v29.h,v30.h,v31.h}[6], [x15]
    st4     {v28.h,v29.h,v30.h,v31.h}[7], [x16]

    b end_func

    direct_save:
    stp  q24,q28, [x0]
    stp	 q25,q29, [x0, 0x20]
    stp	 q26,q30, [x0, 0x40]
    stp	 q27,q31, [x0, 0x60]

end_func:
    ret
        .space  256
        .end

