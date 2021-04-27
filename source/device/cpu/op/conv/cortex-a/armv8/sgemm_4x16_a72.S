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
// 4*16 single precise floating point matric multiplication
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
// optimised for Cortex-A72 pipeline  64 cycle per loop (4*16*4 dot product)
// load 4 more input and 8 more kernel to improve loop performance
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
//         x6 arg6  activation flag     activation layers is integrated after convolution
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
// x6        activation flag
// x9 ~ x10  temp loop counter
// x11~ x13  temp output save address
// x14       output_xy * 4
// x7~8 x15  not used
//
// v0~1 4S data of input0   {i3   i2   i1   i0}
// v2~3 not used
// v4   4S kernal data      {k3 | k2 | k1 | k0}
// v5   4S kernal data      {k7 | k6 | k5 | k4}
// v6   4S kernal data      {kb | ka | k9 | k8}
// v7   4S kernal data      {kf | ke | kd | kc}
// v8~15 not used
// v16 dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 dot product for {i3k3, i2k3, i1k3, i0k3}
// v20 dot product for {i3k4, i2k4, i1k4, i0k4}
// v21 dot product for {i3k5, i2k5, i1k5, i0k5}
// v22 dot product for {i3k6, i2k6, i1k6, i0k6}
// v23 dot product for {i3k7, i2k7, i1k7, i0k7}
// v24 dot product for {i3k8, i2k8, i1k8, i0k8}
// v25 dot product for {i3k9, i2k9, i1k9, i0k9}
// v26 dot product for {i3ka, i2ka, i1ka, i0ka}
// v27 dot product for {i3kb, i2kb, i1kb, i0kb}
// v28 dot product for {i3kc, i2kc, i1kc, i0kc}
// v29 dot product for {i3kd, i2kd, i1kd, i0kd}
// v30 dot product for {i3ke, i2ke, i1ke, i0ke}
// v31 dot product for {i3kf, i2kf, i1kf, i0kf}

    .section .text,"ax"
    .align 5

    .type sgemm_4x16_a72 STT_FUNC
    .global sgemm_4x16_a72
    .hidden sgemm_4x16_a72
sgemm_4x16_a72:
    // bring some code ahead to reduce dependency
    prfm    pldl1keep, [x1]
    cmp     x3, 0x4

    // biases_initial
    cbz     x0, none_biases
    ld4r    {v16.4s,v17.4s,v18.4s,v19.4s}, [x0], 0x10
    ld4r    {v20.4s,v21.4s,v22.4s,v23.4s}, [x0], 0x10
    ld4r    {v24.4s,v25.4s,v26.4s,v27.4s}, [x0], 0x10
    ld4r    {v28.4s,v29.4s,v30.4s,v31.4s}, [x0]
    b       convolution_start

none_biases:
    movi    d16, 0x0
    movi    d17, 0x0
    movi    d18, 0x0
    movi    d19, 0x0
    movi    d20, 0x0
    movi    d21, 0x0
    movi    d22, 0x0
    movi    d23, 0x0
    movi    d24, 0x0
    movi    d25, 0x0
    movi    d26, 0x0
    movi    d27, 0x0
    movi    d28, 0x0
    movi    d29, 0x0
    movi    d30, 0x0
    movi    d31, 0x0

convolution_start:
    ldr     q0, [x1]            // q0=i[3-0]
    ldp     q4, q5, [x2]            // q4=k[3-0] q5=k[7-4] 
    and     x10,x3, 0x3
    lsl     x5, x5, 2           // x5  = output_xy
    b.lt    loop4_end
    lsr     x9, x3, 0x2

// main loop     each loop generate dot prodcut for 4x16x4SP
loop4:  
    fmla    v16.4s, v0.4s,  v4.s[0]     // i[3-0]k[0]
    ldp     q6, q7, [x2, 0x20]      // q6=k[b-8] q7=k[f-c]
    fmla    v17.4s, v0.4s,  v4.s[1]     // i[3-0]k[1]
    subs    x9, x9, 0x1
    fmla    v18.4s, v0.4s,  v4.s[2]     // i[3-0]k[2]
    fmla    v19.4s, v0.4s,  v4.s[3]     // i[3-0]k[3]
    ldr     q1, [x1, 0x10]          // q1=i[3-0]
    fmla    v20.4s, v0.4s,  v5.s[0]     // i[3-0]k[4]
    fmla    v21.4s, v0.4s,  v5.s[1]     // i[3-0]k[5]
    fmla    v22.4s, v0.4s,  v5.s[2]     // i[3-0]k[6]
    fmla    v23.4s, v0.4s,  v5.s[3]     // i[3-0]k[7]
    ldp     q4, q5, [x2, 0x40]      // q4=k[3-0] q5=k[7-4] 
    fmla    v24.4s, v0.4s,  v6.s[0]     // i[3-0]k[8]
    fmla    v25.4s, v0.4s,  v6.s[1]     // i[3-0]k[9]
    fmla    v26.4s, v0.4s,  v6.s[2]     // i[3-0]k[a]
    fmla    v27.4s, v0.4s,  v6.s[3]     // i[3-0]k[b]
    prfm    pldl1keep, [x1, 0x80]
    fmla    v28.4s, v0.4s,  v7.s[0]     // i[3-0]k[c]
    fmla    v29.4s, v0.4s,  v7.s[1]     // i[3-0]k[d]
    prfm    pldl1keep, [x2, 0x140]
    fmla    v30.4s, v0.4s,  v7.s[2]     // i[3-0]k[e]
    fmla    v31.4s, v0.4s,  v7.s[3]     // i[3-0]k[f]

    ldp     q6, q7, [x2, 0x60]      // q6=k[b-8] q7=k[f-c]
    fmla    v16.4s, v1.4s,  v4.s[0]     // i[3-0]k[0]
    fmla    v17.4s, v1.4s,  v4.s[1]     // i[3-0]k[1]
    fmla    v18.4s, v1.4s,  v4.s[2]     // i[3-0]k[2]
    fmla    v19.4s, v1.4s,  v4.s[3]     // i[3-0]k[3]
    ldr     q0, [x1, 0x20]          // q1=i[3-0]
    fmla    v20.4s, v1.4s,  v5.s[0]     // i[3-0]k[4]
    fmla    v21.4s, v1.4s,  v5.s[1]     // i[3-0]k[5]
    fmla    v22.4s, v1.4s,  v5.s[2]     // i[3-0]k[6]
    fmla    v23.4s, v1.4s,  v5.s[3]     // i[3-0]k[7]
    ldp     q4, q5, [x2, 0x80]      // q4=k[3-0] q5=k[7-4] 
    fmla    v24.4s, v1.4s,  v6.s[0]     // i[3-0]k[8]
    fmla    v25.4s, v1.4s,  v6.s[1]     // i[3-0]k[9]
    fmla    v26.4s, v1.4s,  v6.s[2]     // i[3-0]k[a]
    fmla    v27.4s, v1.4s,  v6.s[3]     // i[3-0]k[b]
    prfm    pldl1keep, [x2, 0x180]
    fmla    v28.4s, v1.4s,  v7.s[0]     // i[3-0]k[c]
    fmla    v29.4s, v1.4s,  v7.s[1]     // i[3-0]k[d]
    prfm    pldl1keep, [x2, 0x1c0]
    fmla    v30.4s, v1.4s,  v7.s[2]     // i[3-0]k[e]
    fmla    v31.4s, v1.4s,  v7.s[3]     // i[3-0]k[f]

    ldp     q6, q7, [x2, 0xa0]      // q6=k[b-8] q7=k[f-c]
    fmla    v16.4s, v0.4s,  v4.s[0]     // i[3-0]k[0]
    fmla    v17.4s, v0.4s,  v4.s[1]     // i[3-0]k[1]
    fmla    v18.4s, v0.4s,  v4.s[2]     // i[3-0]k[2]
    fmla    v19.4s, v0.4s,  v4.s[3]     // i[3-0]k[3]
    ldr     q1, [x1, 0x30]          // q1=i[3-0]
    fmla    v20.4s, v0.4s,  v5.s[0]     // i[3-0]k[4]
    fmla    v21.4s, v0.4s,  v5.s[1]     // i[3-0]k[5]
    add     x1, x1, 0x40
    fmla    v22.4s, v0.4s,  v5.s[2]     // i[3-0]k[6]
    fmla    v23.4s, v0.4s,  v5.s[3]     // i[3-0]k[7]
    ldp     q4, q5, [x2, 0xc0]      // q4=k[3-0] q5=k[7-4] 
    fmla    v24.4s, v0.4s,  v6.s[0]     // i[3-0]k[8]
    fmla    v25.4s, v0.4s,  v6.s[1]     // i[3-0]k[9]
    fmla    v26.4s, v0.4s,  v6.s[2]     // i[3-0]k[a]
    fmla    v27.4s, v0.4s,  v6.s[3]     // i[3-0]k[b]
    prfm    pldl1keep, [x2, 0x200]
    fmla    v28.4s, v0.4s,  v7.s[0]     // i[3-0]k[c]
    fmla    v29.4s, v0.4s,  v7.s[1]     // i[3-0]k[d]
    fmla    v30.4s, v0.4s,  v7.s[2]     // i[3-0]k[e]
    fmla    v31.4s, v0.4s,  v7.s[3]     // i[3-0]k[f]

    ldp     q6, q7, [x2, 0xe0]      // q6=k[b-8] q7=k[f-c]
    fmla    v16.4s, v1.4s,  v4.s[0]     // i[3-0]k[0]
    fmla    v17.4s, v1.4s,  v4.s[1]     // i[3-0]k[1]
    add     x2, x2, 0x100
    fmla    v18.4s, v1.4s,  v4.s[2]     // i[3-0]k[2]
    fmla    v19.4s, v1.4s,  v4.s[3]     // i[3-0]k[3]
    ldr     q0, [x1]            // q0=i[3-0]
    fmla    v20.4s, v1.4s,  v5.s[0]     // i[3-0]k[4]
    fmla    v21.4s, v1.4s,  v5.s[1]     // i[3-0]k[5]
    fmla    v22.4s, v1.4s,  v5.s[2]     // i[3-0]k[6]
    fmla    v23.4s, v1.4s,  v5.s[3]     // i[3-0]k[7]
    ldp     q4, q5, [x2]            // q4=k[3-0] q5=k[7-4] 
    fmla    v24.4s, v1.4s,  v6.s[0]     // i[3-0]k[8]
    fmla    v25.4s, v1.4s,  v6.s[1]     // i[3-0]k[9]
    fmla    v26.4s, v1.4s,  v6.s[2]     // i[3-0]k[a]
    fmla    v27.4s, v1.4s,  v6.s[3]     // i[3-0]k[b]
    fmla    v28.4s, v1.4s,  v7.s[0]     // i[3-0]k[c]
    fmla    v29.4s, v1.4s,  v7.s[1]     // i[3-0]k[d]
    fmla    v30.4s, v1.4s,  v7.s[2]     // i[3-0]k[e]
    fmla    v31.4s, v1.4s,  v7.s[3]     // i[3-0]k[f]
    b.ne    loop4


loop4_end:
    lsl     x14,x5, 2           // x14 = output_xy * 4
    cbz     x10, activation

loop1:
    ldp     q6, q7, [x2, 0x20]              // q6=k[b-8] q7=k[f-c]
    fmla    v16.4s, v0.4s,  v4.s[0]     // i[3-0]k[0]
    fmla    v17.4s, v0.4s,  v4.s[1]     // i[3-0]k[1]
    add     x2, x2, 0x40
    fmla    v18.4s, v0.4s,  v4.s[2]     // i[3-0]k[2]
    fmla    v19.4s, v0.4s,  v4.s[3]     // i[3-0]k[3]
    add     x1, x1, 0x10
    fmla    v20.4s, v0.4s,  v5.s[0]     // i[3-0]k[4]
    fmla    v21.4s, v0.4s,  v5.s[1]     // i[3-0]k[5]
    subs    x10, x10 ,0x1
    fmla    v22.4s, v0.4s,  v5.s[2]     // i[3-0]k[6]
    fmla    v23.4s, v0.4s,  v5.s[3]     // i[3-0]k[7]
    ldp     q4, q5, [x2]                    // q4=k[3-0] q5=k[7-4]
    fmla    v24.4s, v0.4s,  v6.s[0]     // i[3-0]k[8]
    fmla    v25.4s, v0.4s,  v6.s[1]     // i[3-0]k[9]
    fmla    v26.4s, v0.4s,  v6.s[2]     // i[3-0]k[a]
    fmla    v27.4s, v0.4s,  v6.s[3]     // i[3-0]k[b]
    fmla    v28.4s, v0.4s,  v7.s[0]     // i[3-0]k[c]
    fmla    v29.4s, v0.4s,  v7.s[1]     // i[3-0]k[d]
    fmla    v30.4s, v0.4s,  v7.s[2]     // i[3-0]k[e]
    fmla    v31.4s, v0.4s,  v7.s[3]     // i[3-0]k[f]
    ldr     q0, [x1]                        // q0=i[3-0]

    b.ne    loop1


activation:
    add     x11,x4, x5          // x11 = output + ouput_xy
    cmp     w6,0
    blt     save_result

    movi    d0, 0
    scvtf   s1,w6
    
    fmax    v16.4s, v16.4s, v0.4s
    fmax    v17.4s, v17.4s, v0.4s
    fmax    v18.4s, v18.4s, v0.4s
    fmax    v19.4s, v19.4s, v0.4s
    fmax    v20.4s, v20.4s, v0.4s
    fmax    v21.4s, v21.4s, v0.4s
    fmax    v22.4s, v22.4s, v0.4s
    fmax    v23.4s, v23.4s, v0.4s
    fmax    v24.4s, v24.4s, v0.4s
    fmax    v25.4s, v25.4s, v0.4s
    fmax    v26.4s, v26.4s, v0.4s
    fmax    v27.4s, v27.4s, v0.4s
    fmax    v28.4s, v28.4s, v0.4s
    fmax    v29.4s, v29.4s, v0.4s
    fmax    v30.4s, v30.4s, v0.4s
    fmax    v31.4s, v31.4s, v0.4s

    beq     save_result
    dup     v1.4s,v1.s[0]

    fmin    v16.4s, v16.4s, v1.4s
    fmin    v17.4s, v17.4s, v1.4s
    fmin    v18.4s, v18.4s, v1.4s
    fmin    v19.4s, v19.4s, v1.4s
    fmin    v20.4s, v20.4s, v1.4s
    fmin    v21.4s, v21.4s, v1.4s
    fmin    v22.4s, v22.4s, v1.4s
    fmin    v23.4s, v23.4s, v1.4s
    fmin    v24.4s, v24.4s, v1.4s
    fmin    v25.4s, v25.4s, v1.4s
    fmin    v26.4s, v26.4s, v1.4s
    fmin    v27.4s, v27.4s, v1.4s
    fmin    v28.4s, v28.4s, v1.4s
    fmin    v29.4s, v29.4s, v1.4s
    fmin    v30.4s, v30.4s, v1.4s
    fmin    v31.4s, v31.4s, v1.4s


save_result:
    add     x12,x4, x5, LSL 1       // x12 = output + ouput_xy * 2
    add     x13,x11,x5, LSL 1       // x13 = output + ouput_xy * 3
    // store result
    // x4 x11 x12 x13 as base address
    cmp     w7,0
    beq    save_result_nchw

    st4     {v16.s,v17.s,v18.s,v19.s}[0], [x4]
    add     x4, x4, 0x10
    st4     {v16.s,v17.s,v18.s,v19.s}[1], [x11]
    add     x11,x11, 0x10
    st4     {v16.s,v17.s,v18.s,v19.s}[2], [x12]
    add     x12,x12, 0x10
    st4     {v16.s,v17.s,v18.s,v19.s}[3], [x13]
    add     x13,x13, 0x10

    st4     {v20.s,v21.s,v22.s,v23.s}[0], [x4]
    add     x4, x4, 0x10
    st4     {v20.s,v21.s,v22.s,v23.s}[1], [x11]
    add     x11,x11, 0x10
    st4     {v20.s,v21.s,v22.s,v23.s}[2], [x12]
    add     x12,x12, 0x10
    st4     {v20.s,v21.s,v22.s,v23.s}[3], [x13]
    add     x13,x13, 0x10

    st4     {v24.s,v25.s,v26.s,v27.s}[0], [x4]
    add     x4, x4, 0x10
    st4     {v24.s,v25.s,v26.s,v27.s}[1], [x11]
    add     x11,x11, 0x10
    st4     {v24.s,v25.s,v26.s,v27.s}[2], [x12]
    add     x12,x12, 0x10
    st4     {v24.s,v25.s,v26.s,v27.s}[3], [x13]
    add     x13,x13, 0x10

    st4     {v28.s,v29.s,v30.s,v31.s}[0], [x4]
    st4     {v28.s,v29.s,v30.s,v31.s}[1], [x11]
    st4     {v28.s,v29.s,v30.s,v31.s}[2], [x12]
    st4     {v28.s,v29.s,v30.s,v31.s}[3], [x13]
    b       end

save_result_nchw:
    str     q16, [x4]
    add     x4, x4, x14
    str     q17, [x11]
    add     x11,x11, x14
    str     q18, [x12]
    add     x12,x12, x14
    str     q19, [x13]
    add     x13,x13, x14

    str     q20, [x4]
    add     x4, x4, x14
    str     q21, [x11]
    add     x11,x11,x14
    str     q22, [x12]
    add     x12,x12,x14
    str     q23, [x13]
    add     x13,x13,x14

    str     q24, [x4]
    add     x4, x4, x14
    str     q25, [x11]
    add     x11,x11,x14
    str     q26, [x12]
    add     x12,x12,x14
    str     q27, [x13]
    add     x13,x13,x14

    str     q28, [x4]
    str     q29, [x11]
    str     q30, [x12]
    str     q31, [x13]

end:

    ret
    .end



