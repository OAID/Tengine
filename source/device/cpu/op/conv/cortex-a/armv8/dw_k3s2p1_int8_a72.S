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
 * Author: haoluo@openailab.com
 */
//      x0: input data  (int 8)
//      x1: kernel data (int 8)
//      x2: output      (int 8)
//      x3: bias data (int 32)
//      x4: out_h
//      x5: out_w
//      x6: multi
//      x7: shift
//      sp: input_w
//      sp + 8: act_min
//      sp + 16: act_max

// input:
//              0  1  2  3  4  5  6  7  8  9
//      line0:  a0 b0 c0 d0 e0 f0 g0 h0 i0 j0
//      line1:  a1 b1 c1 d1 e1 f1 g1 h1 i1 j1
//      line2:  a2 b2 c2 d2 e2 f2 g2 h2 i2 j2
// weight:
//      v20：k00  k01  k02   0   k00   k01   k02   0
//      v21： 0   k00  k01  k02   0    k00   k01  k02
//      v22：k10  k11  k12   0   k10   k11   k12   0
//      v23： 0   k10  k11  k12   0    k10   k11  k12
//      v24：k20  k21  k22   0   k20   k21   k22   0
//      v25： 0   k20  k21  k22   0    k20   k21   k22

// line 0 : v0,v1
// line 1 : v2,v3
// line 2 : v4,v5
// out_scale: 
// int16x8: v6  ~ v9
// not used: v10 ~ v15
// int32x4: v16 ~ v19
// kernel : v20 ~ v25
// out    : v2,v3   
// mutli  : v27
// bias   : v26
// shift  : v28 v29
// relu   : v30 v31

// line 0:-----------smull-------------> int16x8
//      v7 v8 v9 v10
// line 1:-----------smlal-------------> int16x8
//      v7 v8 v9 v10
//  ------------------sadalp-----------> int32x4
//      v16 v17 v18 v19 = saddlp(v7 v8 v9 v10)
// line 2:
//  ------------------smull------------> int16x8
//      v11 v12 v14 v15
//  -----------------sadalp------------> int32x4
//      v16 v17 v18 v19 = sadalp(v11 v12 v14 v15)

// out:
//    v0.4s = addp v7 v8
//    v1.4s = addp v9 v10
// =======================for each line ============================
// step1:
//      a0    b0   c0   d0   e0   f0   g0   h0     ----> out0   out4
//      k00  k01  k02   0   k00   k01  k02  0
//      a0    b0  c0    d0   e0   f0   g0   h0     ----> out1   out5
//      0    k00  k01   k02  0    k00  k01  k02
// step1: extern
//      c0   d0   e0   f0   g0    h0   i0   j0     ----> out2   out6
//      k00  k01  k02   0   k00   k01  k02   0
//      c0   d0    e0    f0  g0   h0   i0   j0     ----> out3   out7
//      0    k00  k01   k02  0    k00  k01  k02
#ifndef KERNEL_NAME
#define KERNEL_NAME depthwise_k3s2p1_int8_a72
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
    sub     sp,sp,0x40
    stp     d8,d9,[sp]
    stp     d10,d11,[sp,0x10]
    stp     d12,d13,[sp,0x20]
    stp     d14,d15,[sp,0x30]

    ldr         x9, [sp, #0x40]     //input_w

    //load the relu value
    ldr         s30,[sp,0x48]
    ldr         s31,[sp,0x50]
    dup         v27.4s, w6          //mutli
    dup         v28.4s, w7          //shift
    dup         v30.4s,v30.s[0]     //min
    dup         v31.4s,v31.s[0]     //max
    //load weight
    ldr     q0, [x1]
    
    //clear the weight vector
    movi d20,0
    movi d21,0
    movi d22,0
    movi d23,0
    movi d24,0
    movi d25,0
    movi d26,0

    smax    v29.4s, v26.4s, v28.4s      //sshl
    smin    v28.4s, v26.4s, v28.4s      //srshl
    cbz     x3, no_bias
    ldr     s26, [x3]
    dup     v26.4s, v26.s[0]
no_bias:

    //init the weight data
    ld1 {v20.b}[0],[x1],#1
    ld1 {v20.b}[1],[x1],#1
    ld1 {v20.b}[2],[x1],#1
    ins v20.b[4],v20.b[0]
    ins v20.b[5],v20.b[1]
    ins v20.b[6],v20.b[2]
    ld1 {v22.b}[0],[x1],#1
    ld1 {v22.b}[1],[x1],#1
    ld1 {v22.b}[2],[x1],#1
    ins v22.b[4],v22.b[0]
    ins v22.b[5],v22.b[1]
    ins v22.b[6],v22.b[2]
    ld1 {v24.b}[0],[x1],#1
    ld1 {v24.b}[1],[x1],#1
    ld1 {v24.b}[2],[x1],#1
    ins v24.b[4],v24.b[0]
    ins v24.b[5],v24.b[1]
    ins v24.b[6],v24.b[2]

    mov     x7, x4
loop_h:
    mov     x10, x0
    add     x11, x0, x9               // input line1
    add     x12, x0, x9, lsl #1       // input line2
    subs    x8, x5, #0x2
    lsr     x6, x8, #0x2
    cmp     x4, #0x1
    beq     last_row
    cmp     x4, x7
    bne     mid_row

first_row_left:
    ldr     q0, [x10]
    ldr     q1, [x11]

    ins     v3.b[0],v22.b[1]
    ins     v3.b[1],v22.b[2]
    ins     v4.b[0],v24.b[1]
    ins     v4.b[1],v24.b[2]

    smull   v6.8h , v0.8b, v3.8b
    smlal   v6.8h , v1.8b, v4.8b
    saddlp  v7.4s, v6.8h
    add     v7.4s,  v7.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s

    st1     {v0.b}[0], [x2], 0x1

    add     x10, x10, #0x1
    add     x11, x11, #0x1


first_row_mid8:
    cmp     x6,#0x0
    beq     first_row_mid4     
    ldr     q0, [x10]                           // v0: a[0,0]--a[0,13]--a[0,15]
    ldr     q1, [x11]                           // v1: a[1,0]--a[1,13]--a[1,15]
    ext     v3.16b, v0.16b, v0.16b, #2          // v3: a[0,2]--a[0,15]--a[0,0],a[0,1]
    ext     v4.16b, v1.16b, v1.16b, #2          // v4: a[1,2]--a[1,15]--a[1,0],a[1,1]
    smull   v6.8h , v0.8b, v22.8b 
    smull   v7.8h , v3.8b, v22.8b 

    saddlp  v16.4s, v6.8h
    saddlp  v17.4s, v7.8h
    
    
    smull   v6.8h , v1.8b, v24.8b       //k2x        a10 x k20, a11 x k21, a12 x k22,     0    , a14 x k20, a15 x k21, a16 x k22,     0
    smull   v7.8h , v4.8b, v24.8b       //k2x        a12 x k20, a13 x k21, a14 x k22,     0    , a16 x k20, a17 x k21, a18 x k22,     0
    
    sadalp  v16.4s, v6.8h
    sadalp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s
    
    st1     {v0.b}[0], [x2], 0x1
    st1     {v0.b}[4], [x2], 0x1
    st1     {v0.b}[8], [x2], 0x1
    st1     {v0.b}[12], [x2], 0x1
    add     x10, x10, #0x8
    add     x11, x11, #0x8
    subs    x6, x6, 0x1
    cmp     x6, #0x0
    bne     first_row_mid8
first_row_mid4:
    and     x6, x8, #0x3
    cmp     x6, #0x0
    beq     first_row_right

    ldr     q0, [x10]
    ldr     q1, [x11]
    ext     v3.16b, v0.16b, v0.16b, #2
    ext     v4.16b, v1.16b, v1.16b, #2
    smull   v6.8h , v0.8b, v22.8b       //k1x  0  4  a00 x k10, a01 x k11, a02 x k12,     0    , a04 x k10, a05 x k11, a06 x k12,     0
    smull   v7.8h , v3.8b, v22.8b       //k1x  2  6  a02 x k10, a03 x k11, a04 x k12,     0    , a06 x k10, a07 x k11, a08 x k12,     0
    
    smlal   v6.8h , v1.8b, v24.8b       //k2x        a10 x k20, a11 x k21, a12 x k22,     0    , a14 x k20, a15 x k21, a16 x k22,     0
    smlal   v7.8h , v4.8b, v24.8b       //k2x        a12 x k20, a13 x k21, a14 x k22,     0    , a16 x k20, a17 x k21, a18 x k22,     0
    
    saddlp  v16.4s, v6.8h               
    saddlp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s

first_row_save_1:
    st1     {v0.b}[0], [x2], 0x1
    ext     v0.16b, v0.16b, v0.16b, #4
    subs    x6, x6, #0x1
    add     x10, x10, #0x2
    add     x11, x11, #0x2
    
    cmp     x6, #0x0
    bne     first_row_save_1
    
first_row_right:
    ldr     q0, [x10]
    ldr     q1, [x11]
    
    ins     v0.b[3],v21.b[0]
    ins     v1.b[3],v21.b[0]

    and     x6, x9, #0x1
    cmp     x6, #0x0
    beq     first_even
    ins     v0.b[2],v21.b[0]
    ins     v1.b[2],v21.b[0]
first_even:
    smull   v6.8h , v0.8b, v22.8b
    smlal   v6.8h , v1.8b, v24.8b
    saddlp  v7.4s, v6.8h

    addp    v6.4s, v7.4s, v7.4s

    add     v7.4s,  v6.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s       
    smin        v0.4s, v0.4s, v31.4s
    st1     {v0.b}[0], [x2], 0x1
    add     x0, x0, x9
    b       loop_h_end


mid_row:
mid_row_left:
    ldr     q0, [x10]
    ldr     q1, [x11]
    ldr     q2, [x12]

    ins     v3.b[0],v20.b[1]
    ins     v3.b[1],v20.b[2]
    ins     v4.b[0],v22.b[1]
    ins     v4.b[1],v22.b[2]
    ins     v5.b[0],v24.b[1]
    ins     v5.b[1],v24.b[2]

    smull   v6.8h , v0.8b, v3.8b
    smlal   v6.8h , v1.8b, v4.8b
    saddlp  v7.4s, v6.8h
    smull   v6.8h , v2.8b, v5.8b
    saddlp  v5.4s, v6.8h
    add     v7.4s,  v7.4s,  v5.4s
    add     v7.4s,  v7.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s       
    smin        v0.4s, v0.4s, v31.4s

    st1     {v0.b}[0], [x2], 0x1
    
    add     x10, x10, #0x1
    add     x11, x11, #0x1
    add     x12, x12, #0x1
    

mid_row_mid8:
    cmp     x6,#0x0
    beq     mid_row_mid4

    ldr     q0, [x10]
    ldr     q1, [x11]
    ldr     q2, [x12]
    ext     v3.16b, v0.16b, v0.16b, #2
    ext     v4.16b, v1.16b, v1.16b, #2
    ext     v5.16b, v2.16b, v2.16b, #2
    smull   v6.8h , v0.8b, v20.8b       //0  4
    smull   v7.8h , v3.8b, v20.8b       //2  6
    
    smlal   v6.8h , v1.8b, v22.8b
    smlal   v7.8h , v4.8b, v22.8b
    
    saddlp  v16.4s, v6.8h
    saddlp  v17.4s, v7.8h

    
    smull   v6.8h , v2.8b, v24.8b
    smull   v7.8h , v5.8b, v24.8b
    sadalp  v16.4s, v6.8h
    sadalp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s
    
    st1     {v0.b}[0], [x2], 0x1
    st1     {v0.b}[4], [x2], 0x1
    st1     {v0.b}[8], [x2], 0x1
    st1     {v0.b}[12], [x2], 0x1
    add     x10, x10, #0x8
    add     x11, x11, #0x8
    add     x12, x12, #0x8
    subs    x6, x6, 0x1
    cmp     x6, #0x0
    bne     mid_row_mid8
mid_row_mid4:
    and     x6, x8, #0x3
    cmp     x6, #0x0
    beq     mid_row_right

    ldr     q0, [x10]
    ldr     q1, [x11]
    ldr     q2, [x12]
    ext     v3.16b, v0.16b, v0.16b, #2
    ext     v4.16b, v1.16b, v1.16b, #2
    ext     v5.16b, v2.16b, v2.16b, #2
    smull   v6.8h , v0.8b, v20.8b       //0  4
    smull   v7.8h , v3.8b, v20.8b       //2  6
    
    smlal   v6.8h , v1.8b, v22.8b
    smlal   v7.8h , v4.8b, v22.8b
    
    saddlp  v16.4s, v6.8h
    saddlp  v17.4s, v7.8h

    
    smull   v6.8h , v2.8b, v24.8b
    smull   v7.8h , v5.8b, v24.8b
    sadalp  v16.4s, v6.8h
    sadalp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s


mid_row_save_1:
    st1     {v0.b}[0], [x2], 0x1
    add     x10, x10, #0x2
    add     x11, x11, #0x2
    add     x12, x12, #0x2
    ext     v0.16b, v0.16b, v0.16b, #4
    subs    x6, x6, #0x1
    bne     mid_row_save_1
    

mid_row_right:
    ldr     q0, [x10]
    ldr     q1, [x11]
    ldr     q2, [x12]

    ins     v0.b[3],v21.b[0]
    ins     v1.b[3],v21.b[0]
    ins     v2.b[3],v21.b[0]

    and     x6, x9, #0x1
    cmp     x6, #0x0
    beq     mid_even
    ins     v0.b[2],v21.b[0]
    ins     v1.b[2],v21.b[0]
    ins     v2.b[2],v21.b[0]
mid_even:

    smull   v6.8h , v0.8b, v20.8b
    smlal   v6.8h , v1.8b, v22.8b
    saddlp  v7.4s, v6.8h
    smull   v6.8h , v2.8b, v24.8b
    saddlp  v5.4s, v6.8h
    add     v7.4s, v7.4s, v5.4s
    addp    v6.4s, v7.4s, v7.4s
    add     v7.4s,  v6.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s       
    smin        v0.4s, v0.4s, v31.4s
    st1     {v0.b}[0], [x2], 0x1
    add     x0, x0, x9,lsl #0x1
    b       loop_h_end

last_row:
    and     x13, x9, #0x1     // assum input_h = input_w
    cmp     x13, #0x0
    beq     mid_row
last_row_left:
    ldr     q0, [x10]
    ldr     q1, [x11]

    ins     v3.b[0],v20.b[1]
    ins     v3.b[1],v20.b[2]
    ins     v4.b[0],v22.b[1]
    ins     v4.b[1],v22.b[2]

    smull   v6.8h , v0.8b, v3.8b
    smlal   v6.8h , v1.8b, v4.8b
    saddlp  v7.4s, v6.8h
    add     v7.4s,  v7.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s       
    smin        v0.4s, v0.4s, v31.4s

    st1     {v0.b}[0], [x2], 0x1

    add     x10, x10, #0x1
    add     x11, x11, #0x1

last_row_mid8:
    cmp     x6,#0x0
    beq     last_row_mid4     
    ldr     q0, [x10]                           // v0: a[0,0]--a[0,13]--a[0,15]
    ldr     q1, [x11]                           // v1: a[1,0]--a[1,13]--a[1,15]
    ext     v3.16b, v0.16b, v0.16b, #2          // v3: a[0,2]--a[0,15]--a[0,0],a[0,1]
    ext     v4.16b, v1.16b, v1.16b, #2          // v4: a[1,2]--a[1,15]--a[1,0],a[1,1]
    smull   v6.8h , v0.8b, v20.8b       
    smull   v7.8h , v3.8b, v20.8b       
    
    smlal   v6.8h , v1.8b, v22.8b       
    smlal   v7.8h , v4.8b, v22.8b        
    
    saddlp  v16.4s, v6.8h               
    saddlp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s
    
    st1     {v0.b}[0], [x2], 0x1
    st1     {v0.b}[4], [x2], 0x1
    st1     {v0.b}[8], [x2], 0x1
    st1     {v0.b}[12], [x2], 0x1
    add     x10, x10, #0x8
    add     x11, x11, #0x8
    subs    x6, x6, 0x1
    cmp     x6, #0x0
    bne     last_row_mid8
last_row_mid4:
    and     x6, x8, #0x3
    cmp     x6, #0x0
    beq     last_row_right

    ldr     q0, [x10]
    ldr     q1, [x11]
    ext     v3.16b, v0.16b, v0.16b, #2
    ext     v4.16b, v1.16b, v1.16b, #2
    smull   v6.8h , v0.8b, v20.8b       
    smull   v7.8h , v3.8b, v20.8b       
    
    smlal   v6.8h , v1.8b, v22.8b       
    smlal   v7.8h , v4.8b, v22.8b    
    
    saddlp  v16.4s, v6.8h               
    saddlp  v17.4s, v7.8h

    addp    v0.4s,  v16.4s, v16.4s    // out0 out2 out0 out2
    addp    v1.4s,  v17.4s, v17.4s    // out1 out3 out1 out3
    zip1    v2.4s,  v0.4s,  v1.4s     // out0 out1 out2 out3

    add     v2.4s,  v2.4s,  v26.4s
    SQRDMULH    v2.4s, v2.4s, v27.4s
    sshl        v2.4s, v2.4s, v29.4s
    srshl       v0.4s, v2.4s, v28.4s
    smax        v0.4s, v0.4s, v30.4s
    smin        v0.4s, v0.4s, v31.4s


last_row_save_1:
    st1     {v0.b}[0], [x2], 0x1
    add     x10, x10, #0x2
    add     x11, x11, #0x2
    add     x12, x12, #0x2
    ext     v0.16b, v0.16b, v0.16b, #4
    subs    x6, x6, #0x1
    bne     last_row_save_1
    
last_row_right:
    ldr     q0, [x10]
    ldr     q1, [x11]
    
    ins     v0.b[3],v21.b[0]
    ins     v1.b[3],v21.b[0]

    and     x6, x9, #0x1
    cmp     x6, #0x0
    beq     first_even
    ins     v0.b[2],v21.b[0]
    ins     v1.b[2],v21.b[0]
last_even:
    smull   v6.8h , v0.8b, v20.8b
    smlal   v6.8h , v1.8b, v22.8b
    saddlp  v7.4s, v6.8h

    addp    v6.4s, v7.4s, v7.4s

    add     v7.4s,  v6.4s,  v26.4s  // add bias
    SQRDMULH    v2.4s, v7.4s, v27.4s    //multi
    sshl        v2.4s, v2.4s, v29.4s    //left shift
    srshl       v0.4s, v2.4s, v28.4s    //right shift
    smax        v0.4s, v0.4s, v30.4s 
    smin        v0.4s, v0.4s, v31.4s
    st1     {v0.b}[0], [x2], 0x1
    b       loop_h_end
    
loop_h_end:
    subs    x4, x4, 0x1
    bne     loop_h


loop_end:
    ldp d8, d9, [sp]
    ldp d10,d11,[sp, 0x10]
    ldp d12,d13,[sp, 0x20]
    ldp d14,d15,[sp, 0x30]
    add sp, sp, 0x40

    ret
