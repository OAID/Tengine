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
// optimised for Cortex-A72 pipeline 14 cycle per loop (4*4*4 dot product)
// this function loads 16 more bytes input data to improve loop performance
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
//         x5 arg5  multi
//         x6 arg6  output xy
//         x7 arg7  shift
//         sp + 4 arg7  act_min
//         sp + 8 arg6  act_max
//
// output: no
//
// register definition
// x0        biases start address
// x1        input start address
// x2        kernel start address
// x3        kernal size
// x4        output start address
// x5        multi
// x6        output_x * output_y
// x7        shift
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
// v24 ~ v31 not used 

        .section .text,"ax"
        .align 5

        .type i8gemm_4x4_a72_int8 STT_FUNC
        .global i8gemm_4x4_a72_int8
        .hidden i8gemm_4x4_a72_int8
i8gemm_4x4_a72_int8:
    // initial
    ldr q0, [x1]        // i3, i2, i1, i0
    cmp x3, 0x4
    movi    d16, 0
    movi    d17, 0
    movi    d18, 0
    movi    d19, 0
    and x11, x3, 0x3

    b.lt    loop4_end
    lsr x9, x3, 2       // x9 = kernel_size / 4

    // main loop     each loop generate dot prodcut for 4x4x4byte
loop4:  
    ldr     q4, [x2]                // k[3-0][1-0]
    rev32   v1.8h, v0.8h        // i2, i3, i0, i1
    prfm    pldl1keep, [x1, 0xc0]
    add     x1, x1, 0x10
    rev64   v2.4s, v0.4s        // i1, i0, i3, i2
    rev64   v3.8h, v0.8h        // i0, i1, i2, i3
    prfm    pldl1keep, [x2, 0xc0]
    add     x2, x2, 0x10
    subs    x9, x9, 1       // loop counter

    smull   v20.8h, v4.8b, v0.8b
    smlal2  v20.8h, v4.16b,v0.16b
    sadalp  v16.4s, v20.8h
    ldr     q0, [x1]        // i3, i2, i1, i0
    smull   v21.8h, v4.8b, v1.8b
    smull   v22.8h, v4.8b, v2.8b
    smull   v23.8h, v4.8b, v3.8b
    smlal2  v21.8h, v4.16b,v1.16b
    smlal2  v22.8h, v4.16b,v2.16b
    smlal2  v23.8h, v4.16b,v3.16b
    sadalp  v17.4s, v21.8h
    sadalp  v18.4s, v22.8h
    sadalp  v19.4s, v23.8h

    b.ne    loop4

loop4_end:
    prfm    pstl1keep, [x4]
    cbz x11, add_bias

// final 2 data
    ldr d4, [x2]                // k[3-0][1-0]
    rev32   v1.4h, v0.4h        // i2, i3, i0, i1
    rev64   v2.2s, v0.2s        // i1, i0, i3, i2
    rev64   v3.4h, v0.4h        // i0, i1, i2, i3

    smull   v20.8h, v4.8b, v0.8b
    sadalp  v16.4s, v20.8h
    smull   v21.8h, v4.8b, v1.8b
    sadalp  v17.4s, v21.8h
    smull   v22.8h, v4.8b, v2.8b
    sadalp  v18.4s, v22.8h
    smull   v23.8h, v4.8b, v3.8b
    sadalp  v19.4s, v23.8h

add_bias:
    cbz     x0, to_int8
    ldr     q0, [x0]
    add     v16.4s,v16.4s,v0.4s
    add     v17.4s,v17.4s,v0.4s
    add     v18.4s,v18.4s,v0.4s
    add     v19.4s,v19.4s,v0.4s

to_int8:
    prfm    pstl1keep, [x4, x6]
    //dup     v0.4s, w5
    //dup     v1.4s, w7
    ldr     q0, [x5]
    ldr     q1, [x7]
    movi    d4, #0
    ldr     s2, [sp]
    ldr     s3, [sp, #0x4]
    dup     v2.4s,v2.s[0]
    dup     v3.4s,v3.s[0]
    smax    v5.4s, v1.4s, v4.4s
    smin    v4.4s, v1.4s, v4.4s
    sqrdmulh    v16.4s, v16.4s, v0.4s
    sqrdmulh    v17.4s, v17.4s, v0.4s
    sqrdmulh    v18.4s, v18.4s, v0.4s
    sqrdmulh    v19.4s, v19.4s, v0.4s
    sshl        v16.4s, v16.4s, v5.4s
    sshl        v17.4s, v17.4s, v5.4s
    sshl        v18.4s, v18.4s, v5.4s
    sshl        v19.4s, v19.4s, v5.4s
    srshl       v16.4s, v16.4s, v4.4s
    srshl       v17.4s, v17.4s, v4.4s
    srshl       v18.4s, v18.4s, v4.4s
    srshl       v19.4s, v19.4s, v4.4s
    
activation:
    add     x1, x4, x6
    smax    v16.4s, v16.4s, v2.4s
    smax    v17.4s, v17.4s, v2.4s
    add     x2, x4, x6, lsl #1
    smax    v18.4s, v18.4s, v2.4s
    smax    v19.4s, v19.4s, v2.4s
    add     x3, x1, x6, lsl #1

    smin    v16.4s, v16.4s, v3.4s
    smin    v17.4s, v17.4s, v3.4s
    smin    v18.4s, v18.4s, v3.4s
    smin    v19.4s, v19.4s, v3.4s

save_result:
    cbz     x6, indirect_save

    st4     {v16.b, v17.b, v18.b, v19.b}[0], [x4]
    st1     {v17.b}[4], [x1], 0x1
    st1     {v16.b}[4], [x1], 0x1
    st1     {v19.b}[4], [x1], 0x1
    st1     {v18.b}[4], [x1]
    st2     {v18.b, v19.b}[8], [x2], 0x2
    st2     {v16.b, v17.b}[8], [x2]
    st1     {v19.b}[12], [x3], 0x1
    st1     {v18.b}[12], [x3], 0x1
    st1     {v17.b}[12], [x3], 0x1
    st1     {v16.b}[12], [x3]
    

    b       save_end

indirect_save:
    stp     q16, q17, [x4]
    stp     q18, q19, [x4, 0x20]

save_end:
    ret
    .end
