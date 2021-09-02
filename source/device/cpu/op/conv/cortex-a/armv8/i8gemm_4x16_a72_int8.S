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
// 4*16 single precise floating point matric multiplication
//
//    --              --      --               --     --               --         --                 --
//    | i0 - - - - - - |      |  k0  k1  ..  kf |     |  b0   b1  .. bf |         | i0k0 i0k1 .. i0kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i1 - - - - - - |      |  .   .   .   .  |     |  b0   b1  .  bf |         | i1k0 i1k1 .. i1kf |
//    |                |  x   |  .   .   .   .  |  +  |                 |     =   |                   |
//    | i2 - - - - - - |      |  .   .   .   .  |     |  b0   b1  .  bf |         | i2k0 i2k1 .. i2kf |
//    |                |      |  .   .   .   .  |     |                 |         |                   |
//    | i3 - - - - - - |      |  .   .   .   .  |     |  b0   b1  .  bf |         | i3k0 i3k1 .. i3kf |
//    --              --      --               --     --               --         --                 --
//      input 4 x p             kernel p x 16            biases 4 x 16                 output 4 x 16           p = kernel size
//
//
// optimised for Cortex-A72 pipeline  41.33 cycle per loop (4*16*4 dot product)
// this function loads 16 more bytes input data to improve loop performance
//
// input:
//         x0 arg0  biases address {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15}     nullptr means no biases
//         x1 arg1  input  address {i[0-3][0-1],i[0-3][2-3],i[0-3][4-5],i[0-3][6-7],...}
//         x2 arg2  kernel address {k[0-15][0-1],k[0-15][2-3],k[0-15][4-5],k[0-15][6-7],...}
//         x3 arg3  kernel size need to be even number
//         x4 arg4  output address 
//                   indirect save:{i0k0,i1k1,i2k2,i3k3, i1k0,i0k1,i3k2,i2k3, i2k0,i3k1,i0k2,i1k3, i3k0,i2k1,i1k2,i0k3}
//                                 {i0k4,i1k5,i2k6,i3k7, i1k4,i0k5,i3k6,i2k7, i2k4,i3k5,i0k6,i1k7, i3k4,i2k5,i1k6,i0k7}
//                                 {i0k8,i1k9,i2ka,i3kb, i1k8,i0k9,i3ka,i2kb, i2k8,i3k9,i0ka,i1kb, i3k8,i2k9,i1ka,i0kb}
//                                 {i0kc,i1kd,i2ke,i3kf, i1kc,i0kd,i3ke,i2kf, i2kc,i3kd,i0ke,i1kf, i3kc,i2kd,i1ke,i0kf}
//                     direct save: output                 : {i0k0  i1k0  i2k0  i3k0}
//                                  output + ouput_xy      : {i0k1  i1k1  i2k1  i3k1}
//                                  output + ouput_xy * 2  : {i0k2  i1k2  i2k2  i3k2}
//                                  ... 
//                                  output + ouput_xy * 15 : {i0k15 i1k15 i2k15 i3k15}
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
// x5        scale address
// x6        output_x * output_y
// x7        activation flag
// x9  ~ x16 temp register
// x17 ~ x31 not used
//
// v0  16byte data of input {i3[3-2], i2[3-2], i1[3-2], i0[3-2], i3[1-0], i2[1-0], i1[1-0], i0[1-0]}
// v1  16byte data of input {i2[3-2], i3[3-2], i0[3-2], i1[3-2], i2[1-0], i3[1-0], i0[1-0], i1[1-0]} = REV32.8H V0
// v2  16byte data of input {i1[3-2], i0[3-2], i3[3-2], i2[3-2], i1[1-0], i0[1-0], i3[1-0], i2[1-0]} = REV64.4S V0
// v3  16byte data of input {i0[3-2], i1[3-2], i2[3-2], i3[3-2], i0[1-0], i1[1-0], i2[1-0], i3[1-0]} = REV64.8H V0
// v4  16byte data of kernel{k3[3-2], k2[3-2], k1[3-2], k0[3-2], k3[1-0], k2[1-0], k1[1-0], k0[1-0]}
// v5  16byte data of kernel{k7[3-2], k6[3-2], k5[3-2], k4[3-2], k7[1-0], k6[1-0], k5[1-0], k4[1-0]}
// v6  16byte data of kernel{kb[3-2], ka[3-2], k9[3-2], k8[3-2], kb[1-0], ka[1-0], k9[1-0], k8[1-0]}
// v7  16byte data of kernel{kf[3-2], ke[3-2], kd[3-2], kc[3-2], kf[1-0], ke[1-0], kd[1-0], kc[1-0]}
// v8  ~ v15 temporary register  must reserved before use
// v16 dot product for {i3k3, i2k2, i1k1, i0k0}   v0 x v4
// v17 dot product for {i2k3, i3k2, i0k1, i1k0}   v1 x v4
// v18 dot product for {i1k3, i0k2, i3k1, i2k0}   v2 x v4
// v19 dot product for {i0k3, i1k2, i2k1, i3k0}   v3 x v4
// v20 dot product for {i3k7, i2k6, i1k5, i0k4}   v0 x v5
// v21 dot product for {i2k7, i3k6, i0k5, i1k4}   v1 x v5
// v22 dot product for {i1k7, i0k6, i3k5, i2k4}   v2 x v5
// v23 dot product for {i0k7, i1k6, i2k5, i3k4}   v3 x v5
// v24 dot product for {i3kb, i2ka, i1k9, i0k8}   v0 x v6
// v25 dot product for {i2kb, i3ka, i0k9, i1k8}   v1 x v6
// v26 dot product for {i1kb, i0ka, i3k9, i2k8}   v2 x v6
// v27 dot product for {i0kb, i1ka, i2k9, i3k8}   v3 x v6
// v28 dot product for {i3kf, i2ke, i1kd, i0kc}   v0 x v7
// v29 dot product for {i2kf, i3ke, i0kd, i1kc}   v1 x v7
// v30 dot product for {i1kf, i0ke, i3kd, i2kc}   v2 x v7
// v31 dot product for {i0kf, i1ke, i2kd, i3kc}   v3 x v7

        .section .text,"ax"
        .align 5

        .type i8gemm_4x16_a72_int8 STT_FUNC
        .global i8gemm_4x16_a72_int8
        .hidden i8gemm_4x16_a72_int8
i8gemm_4x16_a72_int8:
    // initial
    prfm    pldl1keep, [x1]         // preload input data
    movi    d16, 0
    movi    d17, 0
    sub sp, sp, 0x20
    movi    d18, 0
    prfm    pldl1keep, [x2, 0x40]       // preload  kernel data
    movi    d19, 0
    movi    d20, 0
    movi    d21, 0
    movi    d22, 0
    movi    d23, 0
    ld4 {v4.d,v5.d,v6.d,v7.d}[0], [x2]  // load kernel data k[7-0][1-0]
    stp d8, d9, [sp]
    stp d10,d11,[sp, 0x10]
    movi    d24, 0
    cmp x3, 0x4
    movi    d25, 0
    movi    d26, 0
    add x10, x2, 0x20
    movi    d27, 0
    ldr q0, [x1], 0x10          // load input data i0, i1, i2, i3
    movi    d28, 0
    and x11, x3, 0x3
    movi    d29, 0
    movi    d30, 0
    movi    d31, 0

    b.lt    loop4_end
    lsr x9, x3, 2           // x9 = kernel_size / 4

    // main loop     each loop generate dot prodcut for 4x16x4byte
loop4:  
    rev32   v1.8h, v0.8h            // i2, i3, i0, i1
    rev64   v2.4s, v0.4s            // i1, i0, i3, i2
    rev64   v3.8h, v0.8h            // i0, i1, i2, i3
    ld4 {v4.d,v5.d,v6.d,v7.d}[1], [x10] // k[7-0][3-2]
    prfm    pldl1keep, [x2, 0x180]
    add x2, x2, 0x40
    subs    x9, x9, 1           // loop counter

    smull   v8.8h, v4.8b, v0.8b
    smlal2  v8.8h, v4.16b,v0.16b
    sadalp  v16.4s,v8.8h
    smull   v9.8h, v4.8b, v1.8b
    smull   v10.8h,v4.8b, v2.8b
    smlal2  v9.8h, v4.16b,v1.16b
    smlal2  v10.8h,v4.16b,v2.16b
    smull   v11.8h,v4.8b, v3.8b
    sadalp  v17.4s,v9.8h
    smlal2  v11.8h,v4.16b,v3.16b
    smull   v8.8h, v5.8b, v0.8b
    smull   v9.8h, v5.8b, v1.8b
    sadalp  v18.4s,v10.8h
    smull   v10.8h,v5.8b, v2.8b
    sadalp  v19.4s,v11.8h

    smull   v11.8h,v5.8b, v3.8b
    smlal2  v8.8h, v5.16b,v0.16b
    smlal2  v9.8h, v5.16b,v1.16b
    smlal2  v10.8h,v5.16b,v2.16b
    smlal2  v11.8h,v5.16b,v3.16b
    sadalp  v20.4s,v8.8h
    smull   v8.8h, v6.8b, v0.8b
    sadalp  v21.4s,v9.8h
    prfm    pldl1keep, [x1, 0x60]
    smull   v9.8h, v6.8b, v1.8b
    sadalp  v22.4s,v10.8h
    add x10, x10, 0x40
    smull   v10.8h,v6.8b, v2.8b
    sadalp  v23.4s,v11.8h

    smull   v11.8h,v6.8b, v3.8b
    smlal2  v8.8h, v6.16b,v0.16b
    smlal2  v9.8h, v6.16b,v1.16b
    smlal2  v10.8h,v6.16b,v2.16b
    smlal2  v11.8h,v6.16b,v3.16b
    sadalp  v24.4s,v8.8h
    smull   v8.8h, v7.8b, v0.8b
    sadalp  v25.4s,v9.8h
    smull   v9.8h, v7.8b, v1.8b
    sadalp  v26.4s,v10.8h
    smull   v10.8h,v7.8b, v2.8b
    sadalp  v27.4s,v11.8h

    smull   v11.8h,v7.8b, v3.8b
    smlal2  v8.8h, v7.16b,v0.16b
    ldr q0, [x1], 0x10          // i3, i2, i1, i0
    smlal2  v9.8h, v7.16b,v1.16b
    smlal2  v10.8h,v7.16b,v2.16b
    smlal2  v11.8h,v7.16b,v3.16b
    sadalp  v28.4s,v8.8h
    ld4 {v4.d, v5.d, v6.d, v7.d}[0], [x2]  // k[7-0][1-0]
    sadalp  v29.4s,v9.8h
    sadalp  v30.4s,v10.8h
    sadalp  v31.4s,v11.8h

    b.ne    loop4

loop4_end:
    prfm    pstl1keep, [x4]         // preload output channel 0
    cbz x11, add_bias

    // final 2 data
    rev32   v1.4h, v0.4h            // i2, i3, i0, i1
    rev64   v2.2s, v0.2s            // i1, i0, i3, i2
    rev64   v3.4h, v0.4h            // i0, i1, i2, i3

    smull    v8.8h, v4.8b, v0.8b
    sadalp  v16.4s, v8.8h
    smull    v9.8h, v4.8b, v1.8b
    sadalp  v17.4s, v9.8h
    smull   v10.8h, v4.8b, v2.8b
    sadalp  v18.4s,v10.8h
    smull   v11.8h, v4.8b, v3.8b
    sadalp  v19.4s,v11.8h

    smull    v8.8h, v5.8b, v0.8b
    sadalp  v20.4s, v8.8h
    smull    v9.8h, v5.8b, v1.8b
    sadalp  v21.4s, v9.8h
    smull   v10.8h, v5.8b, v2.8b
    sadalp  v22.4s,v10.8h
    smull   v11.8h, v5.8b, v3.8b
    sadalp  v23.4s,v11.8h

    smull    v8.8h, v6.8b, v0.8b
    sadalp  v24.4s, v8.8h
    smull    v9.8h, v6.8b, v1.8b
    sadalp  v25.4s, v9.8h
    smull   v10.8h, v6.8b, v2.8b
    sadalp  v26.4s,v10.8h
    smull   v11.8h, v6.8b, v3.8b
    sadalp  v27.4s,v11.8h

    smull    v8.8h, v7.8b, v0.8b
    sadalp  v28.4s, v8.8h
    smull    v9.8h, v7.8b, v1.8b
    sadalp  v29.4s, v9.8h
    smull   v10.8h, v7.8b, v2.8b
    sadalp  v30.4s,v10.8h
    smull   v11.8h, v7.8b, v3.8b
    sadalp  v31.4s,v11.8h

add_bias:
    prfm    pstl1keep, [x4, x6]         // preload output channel 0
    cbz     x0, to_int8
    ldr     q0, [x0]
    ldr     q1, [x0, 0x10]
    add     v16.4s,v16.4s,v0.4s
    add     v17.4s,v17.4s,v0.4s
    ldr     q2, [x0, 0x20]
    add     v18.4s,v18.4s,v0.4s
    add     v19.4s,v19.4s,v0.4s
    ldr     q3, [x0, 0x30]

    add     v20.4s,v20.4s,v1.4s
    add     v21.4s,v21.4s,v1.4s
    add     v22.4s,v22.4s,v1.4s
    add     v23.4s,v23.4s,v1.4s

    add     v24.4s,v24.4s,v2.4s
    add     v25.4s,v25.4s,v2.4s
    add     v26.4s,v26.4s,v2.4s
    add     v27.4s,v27.4s,v2.4s

    add     v28.4s,v28.4s,v3.4s
    add     v29.4s,v29.4s,v3.4s
    add     v30.4s,v30.4s,v3.4s
    add     v31.4s,v31.4s,v3.4s

to_int8:
    add     x1, x4, x6
    prfm    pstl1keep, [x4, x6]
    //dup     v0.4s, w5
    //dup     v1.4s, w7
    ldr     q0, [x5]
    ldr     q1, [x7]
    movi    d4, #0
    ldr     s2, [sp, #0x20]
    ldr     s3, [sp, #0x28]
    dup     v2.4s,v2.s[0]
    dup     v3.4s,v3.s[0]
    smax    v5.4s, v1.4s, v4.4s
    smin    v4.4s, v1.4s, v4.4s

    sqrdmulh    v16.4s, v16.4s, v0.4s
    sqrdmulh    v17.4s, v17.4s, v0.4s
    sqrdmulh    v18.4s, v18.4s, v0.4s
    sqrdmulh    v19.4s, v19.4s, v0.4s

    ldr     q0, [x5, 0x10]
    add     x2, x4, x6, lsl #1
    sqrdmulh     v20.4s,v20.4s,v0.4s
    sqrdmulh     v21.4s,v21.4s,v0.4s
    add     x3, x1, x6, lsl #1
    sqrdmulh     v22.4s,v22.4s,v0.4s
    sqrdmulh     v23.4s,v23.4s,v0.4s
    add     x9, x4, x6, lsl #2
    add     x10, x1, x6, lsl #2
    add     x11, x2, x6, lsl #2
    add     x12, x3, x6, lsl #2

    ldr     q0, [x5, 0x20]
    sqrdmulh     v24.4s,v24.4s,v0.4s
    sqrdmulh     v25.4s,v25.4s,v0.4s
    sqrdmulh     v26.4s,v26.4s,v0.4s
    sqrdmulh     v27.4s,v27.4s,v0.4s

    ldr     q0, [x5, 0x30]
    sqrdmulh     v28.4s,v28.4s,v0.4s
    sqrdmulh     v29.4s,v29.4s,v0.4s
    sqrdmulh     v30.4s,v30.4s,v0.4s
    sqrdmulh     v31.4s,v31.4s,v0.4s

    sshl        v16.4s, v16.4s, v5.4s
    sshl        v17.4s, v17.4s, v5.4s
    sshl        v18.4s, v18.4s, v5.4s
    sshl        v19.4s, v19.4s, v5.4s
    srshl       v16.4s, v16.4s, v4.4s
    srshl       v17.4s, v17.4s, v4.4s
    srshl       v18.4s, v18.4s, v4.4s
    srshl       v19.4s, v19.4s, v4.4s

    ldr     q1, [x7, 0x10]
    movi    d4, #0
    smax    v5.4s, v1.4s, v4.4s
    smin    v4.4s, v1.4s, v4.4s
    sshl        v20.4s, v20.4s, v5.4s
    sshl        v21.4s, v21.4s, v5.4s
    sshl        v22.4s, v22.4s, v5.4s
    sshl        v23.4s, v23.4s, v5.4s
    srshl       v20.4s, v20.4s, v4.4s
    srshl       v21.4s, v21.4s, v4.4s
    srshl       v22.4s, v22.4s, v4.4s
    srshl       v23.4s, v23.4s, v4.4s

    ldr     q1, [x7, 0x20]
    movi    d4, #0
    smax    v5.4s, v1.4s, v4.4s
    smin    v4.4s, v1.4s, v4.4s
    sshl        v24.4s, v24.4s, v5.4s
    sshl        v25.4s, v25.4s, v5.4s
    sshl        v26.4s, v26.4s, v5.4s
    sshl        v27.4s, v27.4s, v5.4s
    srshl       v24.4s, v24.4s, v4.4s
    srshl       v25.4s, v25.4s, v4.4s
    srshl       v26.4s, v26.4s, v4.4s
    srshl       v27.4s, v27.4s, v4.4s

    ldr     q1, [x7, 0x30]
    movi    d4, #0
    smax    v5.4s, v1.4s, v4.4s
    smin    v4.4s, v1.4s, v4.4s
    sshl        v28.4s, v28.4s, v5.4s
    sshl        v29.4s, v29.4s, v5.4s
    sshl        v30.4s, v30.4s, v5.4s
    sshl        v31.4s, v31.4s, v5.4s
    srshl       v28.4s, v28.4s, v4.4s
    srshl       v29.4s, v29.4s, v4.4s
    srshl       v30.4s, v30.4s, v4.4s
    srshl       v31.4s, v31.4s, v4.4s
    
activation:
    smax    v16.4s, v16.4s, v2.4s
    smax    v17.4s, v17.4s, v2.4s
    smax    v18.4s, v18.4s, v2.4s
    smax    v19.4s, v19.4s, v2.4s
    smax     v20.4s,v20.4s,v2.4s
    smax     v21.4s,v21.4s,v2.4s
    smax     v22.4s,v22.4s,v2.4s
    smax     v23.4s,v23.4s,v2.4s

    smax     v24.4s,v24.4s,v2.4s
    smax     v25.4s,v25.4s,v2.4s
    smax     v26.4s,v26.4s,v2.4s
    smax     v27.4s,v27.4s,v2.4s

    smax     v28.4s,v28.4s,v2.4s
    smax     v29.4s,v29.4s,v2.4s
    smax     v30.4s,v30.4s,v2.4s
    smax     v31.4s,v31.4s,v2.4s

    smin    v16.4s, v16.4s, v3.4s
    smin    v17.4s, v17.4s, v3.4s
    smin    v18.4s, v18.4s, v3.4s
    smin    v19.4s, v19.4s, v3.4s
    smin     v20.4s,v20.4s,v3.4s
    smin     v21.4s,v21.4s,v3.4s
    smin     v22.4s,v22.4s,v3.4s
    smin     v23.4s,v23.4s,v3.4s

    smin     v24.4s,v24.4s,v3.4s
    smin     v25.4s,v25.4s,v3.4s
    smin     v26.4s,v26.4s,v3.4s
    smin     v27.4s,v27.4s,v3.4s

    smin     v28.4s,v28.4s,v3.4s
    smin     v29.4s,v29.4s,v3.4s
    smin     v30.4s,v30.4s,v3.4s
    smin     v31.4s,v31.4s,v3.4s

save_result_nhwc:
    cbz     x6, indirect_save

    add     x13, x4, x6, LSL 3       // channel8
    add     x14, x1, x6, LSL 3       // channel9
    add     x15, x2, x6, LSL 3       // channel10
    add     x16, x3, x6, LSL 3       // channel11
// v16 dot product for {i3k3, i2k2, i1k1, i0k0}   v0 x v4
// v17 dot product for {i2k3, i3k2, i0k1, i1k0}   v1 x v4
// v18 dot product for {i1k3, i0k2, i3k1, i2k0}   v2 x v4
// v19 dot product for {i0k3, i1k2, i2k1, i3k0}   v3 x v4
// v20 dot product for {i3k7, i2k6, i1k5, i0k4}   v0 x v5
// v21 dot product for {i2k7, i3k6, i0k5, i1k4}   v1 x v5
// v22 dot product for {i1k7, i0k6, i3k5, i2k4}   v2 x v5
// v23 dot product for {i0k7, i1k6, i2k5, i3k4}   v3 x v5
// v24 dot product for {i3kb, i2ka, i1k9, i0k8}   v0 x v6
// v25 dot product for {i2kb, i3ka, i0k9, i1k8}   v1 x v6
// v26 dot product for {i1kb, i0ka, i3k9, i2k8}   v2 x v6
// v27 dot product for {i0kb, i1ka, i2k9, i3k8}   v3 x v6
// v28 dot product for {i3kf, i2ke, i1kd, i0kc}   v0 x v7
// v29 dot product for {i2kf, i3ke, i0kd, i1kc}   v1 x v7
// v30 dot product for {i1kf, i0ke, i3kd, i2kc}   v2 x v7
// v31 dot product for {i0kf, i1ke, i2kd, i3kc}   v3 x v7

    st4     {v16.b, v17.b, v18.b, v19.b}[0], [x4]
    st1     {v17.b}[4], [x1], 0x1
    st1     {v16.b}[4], [x1], 0x1
    st1     {v19.b}[4], [x1], 0x1
    st1     {v18.b}[4], [x1]
    st2     {v18.b, v19.b}[8], [x2], 0x2
    st2     {v16.b, v17.b}[8], [x2]
    st1     {v19.b}[12], [x3], 0x1
    add     x0, x9,  x6, LSL 3
    st1     {v18.b}[12], [x3], 0x1
    add     x1, x10, x6, LSL 3
    st1     {v17.b}[12], [x3], 0x1
    add     x2, x11, x6, LSL 3
    st1     {v16.b}[12], [x3]
    add     x3, x12, x6, LSL 3

    st4     {v20.b, v21.b, v22.b, v23.b}[0], [x9]
    st1     {v21.b}[4], [x10], 0x1
    st1     {v20.b}[4], [x10], 0x1
    st1     {v23.b}[4], [x10], 0x1
    st1     {v22.b}[4], [x10]
    st2     {v22.b, v23.b}[8], [x11], 0x2
    st2     {v20.b, v21.b}[8], [x11]
    st1     {v23.b}[12], [x12], 0x1
    st1     {v22.b}[12], [x12], 0x1
    st1     {v21.b}[12], [x12], 0x1
    st1     {v20.b}[12], [x12]

    st4     {v24.b, v25.b, v26.b, v27.b}[0], [x13]
    st1     {v25.b}[4], [x14], 0x1
    st1     {v24.b}[4], [x14], 0x1
    st1     {v27.b}[4], [x14], 0x1
    st1     {v26.b}[4], [x14]
    st2     {v26.b, v27.b}[8], [x15], 0x2
    st2     {v24.b, v25.b}[8], [x15]
    st1     {v27.b}[12], [x16], 0x1
    st1     {v26.b}[12], [x16], 0x1
    st1     {v25.b}[12], [x16], 0x1
    st1     {v24.b}[12], [x16]

    st4     {v28.b, v29.b, v30.b, v31.b}[0], [x0]
    st1     {v29.b}[4], [x1], 0x1
    st1     {v28.b}[4], [x1], 0x1
    st1     {v31.b}[4], [x1], 0x1
    st1     {v30.b}[4], [x1]
    st2     {v30.b, v31.b}[8], [x2], 0x2
    st2     {v28.b, v29.b}[8], [x2]
    st1     {v31.b}[12], [x3], 0x1
    st1     {v30.b}[12], [x3], 0x1
    st1     {v29.b}[12], [x3], 0x1
    st1     {v28.b}[12], [x3]


    b       save_end
indirect_save:
    stp     q16, q17, [x4]
    stp     q18, q19, [x4, 0x20]
    stp     q20, q21, [x4, 0x40]
    stp     q22, q23, [x4, 0x60]
    stp     q24, q25, [x4, 0x80]
    stp     q26, q27, [x4, 0xa0]
    stp     q28, q29, [x4, 0xc0]
    stp     q30, q31, [x4, 0xe0]

save_end:
    ldp d8,  d9,  [sp]
    ldp d10, d11, [sp, 0x10]
    add sp, sp, 0x20

    ret

        .end
