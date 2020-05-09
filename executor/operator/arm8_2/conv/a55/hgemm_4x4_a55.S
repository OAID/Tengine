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
// 4*4 half precise floating point matric multiplication
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
// optimised for Cortex-A55 pipeline 24 cycle per loop (4*4*4 dot product)
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
// x6        fused relu flag
// x9 ~ x10  temp loop counter
// x11~ x13  temp output save address
// x7~8 14~15 ot used

//
// v0-3 4h data of input0   {i3   i2   i1   i0}
// v4   8h kernal data      {k3 | k2 | k1 | k0 | k3 | k2 | k1 | k0}
// v5   8h kernal data      {k3 | k2 | k1 | k0 | k3 | k2 | k1 | k0}
// v8~v15 not used
// v16 4h dot product for {i3k0, i2k0, i1k0, i0k0}
// v17 4h dot product for {i3k1, i2k1, i1k1, i0k1}
// v18 4h dot product for {i3k2, i2k2, i1k2, i0k2}
// v19 4h dot product for {i3k3, i2k3, i1k3, i0k3}
// v20~V31 not used

        .section .text,"ax"
        .align 5

        .type hgemm_4x4_a55 STT_FUNC
        .global hgemm_4x4_a55
        .hidden hgemm_4x4_a55
hgemm_4x4_a55:
        // bring some code ahead to reduce dependency
        cmp     x3, 0x4

        // biases_initial
        cbz     x0, none_biases
        ld4r    {v16.4h,v17.4h,v18.4h,v19.4h}, [x0]
        b       convolution_start

none_biases:
        movi    d16, 0x0
        movi    d17, 0x0
        movi    d18, 0x0
        movi    d19, 0x0

convolution_start:
        and     x10,x3, 0x3
        lsl     x5, x5, 1                       // x5  = output_xy
        b.lt    loop4_end
        lsr     x9, x3, 0x2

// main loop     each loop generate dot prodcut for 4x4x4 half precise data
loop4:
        ldp     q4, q5, [x2]                    // q4=k[3-0][1-0] q5=k[3-0][3-2]
        ldp     d0, d1, [x1]                    // d0=i[3-0][0] d1=i[3-0][1]
        ldp     d2, d3, [x1, 0x10]              // d2=i[3-0][2] d3=i[3-0][3]
        subs    x9, x9, 0x1
        fmla    v16.4h, v0.4h,  v4.h[0]         // i[3-0]k[0]
        fmla    v17.4h, v0.4h,  v4.h[1]         // i[3-0]k[1]
        fmla    v18.4h, v0.4h,  v4.h[2]         // i[3-0]k[2]
        fmla    v19.4h, v0.4h,  v4.h[3]         // i[3-0]k[3]

        fmla    v16.4h, v1.4h,  v4.h[4]         // i[3-0]k[0]
        fmla    v17.4h, v1.4h,  v4.h[5]         // i[3-0]k[1]
        fmla    v18.4h, v1.4h,  v4.h[6]         // i[3-0]k[2]
        fmla    v19.4h, v1.4h,  v4.h[7]         // i[3-0]k[3]
       
        fmla    v16.4h, v2.4h,  v5.h[0]         // i[3-0]k[0]
        fmla    v17.4h, v2.4h,  v5.h[1]         // i[3-0]k[1]
        add     x2, x2, 0x20
        fmla    v18.4h, v2.4h,  v5.h[2]         // i[3-0]k[2]
        fmla    v19.4h, v2.4h,  v5.h[3]         // i[3-0]k[3]
        add     x1, x1, 0x20

        fmla    v16.4h, v3.4h,  v5.h[4]         // i[3-0]k[0]
        fmla    v17.4h, v3.4h,  v5.h[5]         // i[3-0]k[1]
        fmla    v18.4h, v3.4h,  v5.h[6]         // i[3-0]k[2]
        fmla    v19.4h, v3.4h,  v5.h[7]         // i[3-0]k[3]
        b.ne    loop4


loop4_end:
        add     x11,x4, x5                      // x11 = output + ouput_xy
        cbz     x10, fused_relu

loop1:
        ldr     d0, [x1], 0x8                   // d0=i[3-0]
        ldr     d4, [x2], 0x8                   // q4=k[3-0]
        subs    x10, x10 ,0x1
        fmla    v16.4h, v0.4h,  v4.h[0]         // i[3-0]k[0]
        fmla    v17.4h, v0.4h,  v4.h[1]         // i[3-0]k[1]
        fmla    v18.4h, v0.4h,  v4.h[2]         // i[3-0]k[2]
        fmla    v19.4h, v0.4h,  v4.h[3]         // i[3-0]k[3]

        b.ne    loop1


fused_relu:
        add     x12,x4, x5, LSL 1               // x12 = output + ouput_xy * 2
        add     x13,x11,x5, LSL 1               // x13 = output + ouput_xy * 3
        cmp     x6, 0
        blt     save_result

        movi    d0, 0x0
        scvtf   h1, x6
        fmax    v16.4h, v16.4h, v0.4h
        fmax    v17.4h, v17.4h, v0.4h
        fmax    v18.4h, v18.4h, v0.4h
        fmax    v19.4h, v19.4h, v0.4h
            
        beq     save_result
        dup     v1.4h, v1.h[0]
        fmin    v16.4h, v16.4h, v1.4h
        fmin    v17.4h, v17.4h, v1.4h
        fmin    v18.4h, v18.4h, v1.4h
        fmin    v19.4h, v19.4h, v1.4h

save_result:

        // store result
        // x4 x11 x12 x13 as base address   x14 = output_xy * 4
        str     d16, [x4]
        str     d17, [x11]
        str     d18, [x12]
        str     d19, [x13]

        ret

        .end

