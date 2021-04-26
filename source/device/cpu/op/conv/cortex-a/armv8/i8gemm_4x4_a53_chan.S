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
 * Copyright (c) 2017, Open AI Lab
 * Author: xiaowei@openailab.com
 */
//
// 4*4 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  k3 |     |  b0  b1  b2  b3 |         | i0k0 i0k1 .. i0k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i1k0 i1k1 .. i1k3 |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i2k0 i2k1 .. i2k3 |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0  b1  b2  b3 |         | i3k0 i3k1 .. i3k3 |
//    --              --      --               --     --               --         --                 --
//      input 4 x p             kernel p x 4             biases 4 x 4                 output 4 x 16           p = kernel size
//
//
// optimised for Cortex-A53 pipeline 23 cycle per loop (4*4*4 dot product)
//
// input:
//         x0 arg0  biases address {b0, b1, b2, b3}
//         x1 arg1  input  address {i[0-3][0-1],i[0-3][2-3],i[0-3][4-5],i[0-3][6-7],i[0-3][8-9],...}
//         x2 arg2  kernel address {k[0-3][0-1],k[0-3][2-3],k[0-3][4-5],k[0-3][6-7],...}
//         x3 arg3  kernel size need to be even number
//         x4 arg4  output address 
//                   indirect save:{i0k0,i1k1,i2k2,i3k3, i1k0,i0k1,i3k2,i2k3, i2k0,i3k1,i0k2,i1k3, i3k0,i2k1,i1k2,i0k3}
//                     direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                  output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                  output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                  output + ouput_xy * 3  : {i0k3  i1k3  i2k3  i3k3}
//         x5 arg5  scale address
//         x6 arg6  output xy
//         x7 arg7  activation flag  relu layers is integrated after convolution
//
// output: no
//
// register definition
// x0        biases start address
// x1        input start address
// x2        kernel start address
// x3        kernal size
// x4        output start address
// x5        scale address
// x6        output_x * output_y
// x7        activation flag
// x9        temp loop counter
// x8 ~ x31  not used
//
// v0  16byte data of input {i3[3-2], i2[3-2], i1[3-2], i0[3-2], i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// v1  16byte data of input {i2[3-2], i3[3-2], i0[3-2], i1[3-2], i2[1-0], i3[1-0], i0[1-0], i1[1-0]} = REV32.8H V0
// v2  16byte data of input {i1[3-2], i0[3-2], i3[3-2], i2[3-2], i1[1-0], i0[1-0], i3[1-0], i2[1-0]} = REV64.4S V0
// v3  16byte data of input {i0[3-2], i1[3-2], i2[3-2], i3[3-2], i0[1-0], i1[1-0], i2[1-0], i3[1-0]} = REV64.8H V0
// v4  16byte data of kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2], k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// v5 
// v6  
// v7  
// v8  ~ v15 not used
// v16 dot product for {i3k3, i2k2, i1k1, i0k0}   v0 x v4
// v17 dot product for {i2k3, i3k2, i0k1, i1k0}   v1 x v4
// v18 dot product for {i1k3, i0k2, i3k1, i2k0}   v2 x v4
// v19 dot product for {i0k3, i1k2, i2k1, i3k0}   v3 x v4
// v20 ~ v23 temp register
// v24 ~ v31  not used 

    .section .text,"ax"
    .align 5

    .type i8gemm_4x4_a53_chan STT_FUNC
    .global i8gemm_4x4_a53_chan
    .hidden i8gemm_4x4_a53_chan
i8gemm_4x4_a53_chan:
    // initial
    cmp x3, 0x4
    movi    d16, 0
    movi    d17, 0
    lsl x6,  x6,  0x2
    movi    d18, 0
    movi    d19, 0
    and x11, x3, 0x3

    b.lt    loop4_end
    lsr x9, x3, 2       // x9 = kernel_size / 4

    // main loop     each loop generate dot prodcut for 4x4x4byte
loop4:  
    ldr q0, [x1]        // i3, i2, i1, i0
    ldr q4, [x2]                // k[3-0][1-0]
    rev32   v1.8h, v0.8h        // i2, i3, i0, i1
    rev64   v2.4s, v0.4s        // i1, i0, i3, i2
    rev64   v3.8h, v0.8h        // i0, i1, i2, i3

    smull   v20.8h, v4.8b, v0.8b
    subs    x9, x9, 1       // loop counter
    smull   v21.8h, v4.8b, v1.8b
    smull   v22.8h,v4.8b, v2.8b
    smull   v23.8h,v4.8b, v3.8b
    smlal2  v20.8h, v4.16b,v0.16b
    prfm    pldl1keep, [x1, 0xc0]
    smlal2  v21.8h, v4.16b,v1.16b
    prfm    pldl1keep, [x2, 0xc0]
    smlal2  v22.8h,v4.16b,v2.16b
    add     x1, x1, 0x10
    smlal2  v23.8h,v4.16b,v3.16b
    add     x2, x2, 0x10
    sadalp  v16.4s,v20.8h
    sadalp  v17.4s,v21.8h
    sadalp  v18.4s,v22.8h
    sadalp  v19.4s,v23.8h

    b.ne    loop4

loop4_end:
    prfm    pstl1keep, [x4]
    cbz     x11, int2fp

// final 2 data
    ldr     d0, [x1]        // i3, i2, i1, i0
    ldr     d4, [x2]                // k[3-0][1-0]
    rev32   v1.4h, v0.4h        // i2, i3, i0, i1
    rev64   v2.2s, v0.2s        // i1, i0, i3, i2
    rev64   v3.4h, v0.4h        // i0, i1, i2, i3

    smull   v20.8h, v4.8b, v0.8b
    smull   v21.8h, v4.8b, v1.8b
    smull   v22.8h, v4.8b, v2.8b
    smull   v23.8h, v4.8b, v3.8b
    sadalp  v16.4s, v20.8h
    sadalp  v17.4s, v21.8h
    sadalp  v18.4s, v22.8h
    sadalp  v19.4s, v23.8h

int2fp:
    // convert result to sp and multiply with scale
    ldr     q0, [x5]
    scvtf   v16.4s, v16.4s
    prfm    pstl1keep, [x4, x6]
    scvtf   v17.4s, v17.4s
    add     x1, x4, x6
    scvtf   v18.4s, v18.4s
    add     x2, x4, x6, LSL 1
    scvtf   v19.4s, v19.4s
    fmul    v16.4s, v16.4s, v0.4s
    add     x3, x2, x6 
    fmul    v17.4s, v17.4s, v0.4s
    fmul    v18.4s, v18.4s, v0.4s
    fmul    v19.4s, v19.4s, v0.4s

    cbz x0, activation

// load and add biases
    ldr     q0, [x0]
    fadd    v16.4s,v16.4s,v0.4s
    fadd    v17.4s,v17.4s,v0.4s
    fadd    v18.4s,v18.4s,v0.4s
    fadd    v19.4s,v19.4s,v0.4s

activation:
    cmp      w7,0
    blt      save_result

    movi    d0, 0x0
    scvtf   s1, w7

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
    cbz     x6, indirect_save

    ldr     w7, [sp]
    cbz     w7, save_result_nchw

    st1     {v16.s}[0], [x4], 0x4
    st1     {v17.s}[1], [x4], 0x4
    st1     {v18.s}[2], [x4], 0x4
    st1     {v19.s}[3], [x4]
    st1     {v17.s}[0], [x1], 0x4
    st1     {v16.s}[1], [x1], 0x4
    st1     {v19.s}[2], [x1], 0x4
    st1     {v18.s}[3], [x1]
    st1     {v18.s}[0], [x2], 0x4
    st1     {v19.s}[1], [x2], 0x4
    st1     {v16.s}[2], [x2], 0x4
    st1     {v17.s}[3], [x2]
    st1     {v19.s}[0], [x3], 0x4
    st1     {v18.s}[1], [x3], 0x4
    st1     {v17.s}[2], [x3], 0x4
    st1     {v16.s}[3], [x3]
    b       save_end

save_result_nchw:
    st4     {v16.s, v17.s, v18.s, v19.s}[0], [x4]
    st1     {v17.s}[1], [x1], 0x4
    st1     {v19.s}[3], [x3], 0x4
    st1     {v16.s}[1], [x1], 0x4
    st1     {v18.s}[3], [x3], 0x4
    st1     {v19.s}[1], [x1], 0x4
    st1     {v17.s}[3], [x3], 0x4
    st2     {v18.s, v19.s}[2], [x2], 0x8
    st1     {v18.s}[1], [x1]
    st2     {v16.s, v17.s}[2], [x2]
    st1     {v16.s}[3], [x3]

    b       save_end

indirect_save:
    stp     q16, q17, [x4]
    stp     q18, q19, [x4, 0x20]

save_end:
    ret

        .end
