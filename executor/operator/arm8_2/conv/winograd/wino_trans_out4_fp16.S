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
 * Author: chunyinglv@openailab.com
*/


//x0: buffer
//x1: output
//x2: out_hw*sizeof(float)
//x3: ker
//x4: bias
//x5: activation


// r1+ r2: v20 - v25
// r1- r2: v26 -v31
// r3+ r4: v8 - v13
//(r3-r4)*2: v14- v19
    .section .text,"ax"
    .align 5

    .type wino_trans_out4_fp16 STT_FUNC
    .global wino_trans_out4_fp16
    .hidden wino_trans_out4_fp16
    
wino_trans_out4_fp16:
    sub    sp, sp, 0x40
    stp    d8, d9, [sp]
    stp    d10,d11,[sp, 0x10]
    stp    d12,d13,[sp, 0x20]
    stp    d14,d15,[sp, 0x30]
    
comput_idx:
    //str[x1,x11,x12,x13]
    add x11,x1,x2      
    add x12,x1,x2,LSL 1     
    add    x13,x11,x2, LSL 1  
    //ldr[x0,x8,x9,x10]
    add x8,x0,#0x30                  //line1: x0 + 6*4*sizeof(fp16)=0x30
    add x9,x0,#0x90                  //line3: 0x30 * 3
    add x10,x0,#0xf0                //line5: 0x30 * 5

load:
    //load v0-v11
    //add:v20-v25
    //sub:v26-v31
    
    ldr d0, [x8]
    ldr d1, [x8,#0x8]
    ldr d2, [x8,#0x10]
    ldr d3, [x8,#0x18]
    ldr d4, [x8,#0x20]
    ldr d5, [x8,#0x28]
    ldr d6, [x8,#0x30]
    ldr d7, [x8,#0x38]
    ldr d8, [x8,#0x40]
    ldr d9, [x8,#0x48]
    ldr d10, [x8,#0x50]
    ldr d11, [x8,#0x58]


    fadd v20.4h,v0.4h,v6.4h
    fadd v21.4h,v1.4h,v7.4h
    fadd v22.4h,v2.4h,v8.4h
    fadd v23.4h,v3.4h,v9.4h
    fadd v24.4h,v4.4h,v10.4h
    fadd v25.4h,v5.4h,v11.4h
    
    fsub v26.4h,v0.4h,v6.4h
    fsub v27.4h,v1.4h,v7.4h
    fsub v28.4h,v2.4h,v8.4h
    fsub v29.4h,v3.4h,v9.4h
    fsub v30.4h,v4.4h,v10.4h
    fsub v31.4h,v5.4h,v11.4h

    //load:v0-v7
    //add:v8-v13
    //sub:v14-v19

    ldr d0, [x9]
    ldr d1, [x9,#0x8]
    ldr d2, [x9,#0x10]
    ldr d3, [x9,#0x18]

    ldr d4, [x9,#0x30]
    ldr d5, [x9,#0x38]
    ldr d6, [x9,#0x40]
    ldr d7, [x9,#0x48]

    fadd v8.4h,v0.4h,v4.4h
    fadd v9.4h,v1.4h,v5.4h
    fadd v10.4h,v2.4h,v6.4h
    fadd v11.4h,v3.4h,v7.4h
  
    fsub v14.4h,v0.4h,v4.4h
    fsub v15.4h,v1.4h,v5.4h
    fsub v16.4h,v2.4h,v6.4h
    fsub v17.4h,v3.4h,v7.4h

    ldr d0, [x9,#0x20]
    ldr d1, [x9,#0x28]
    ldr d2, [x9,#0x50]
    ldr d3, [x9,#0x58]

    fadd v12.4h,v0.4h,v2.4h
    fadd v13.4h,v1.4h,v3.4h
    fsub v18.4h,v0.4h,v2.4h
    fsub v19.4h,v1.4h,v3.4h

    ldr q0,[x3]

    fmul v14.4h,v14.4h,v0.h[0]
    fmul v15.4h,v15.4h,v0.h[0]
    fmul v16.4h,v16.4h,v0.h[0]
    fmul v17.4h,v17.4h,v0.h[0]
    fmul v18.4h,v18.4h,v0.h[0]
    fmul v19.4h,v19.4h,v0.h[0]

//line1: mid[v1,v4,v5,v6,v7,v(4)]
line1:
    //(r1-r2)+2*(r3-r4)
    fadd v1.4h,v26.4h,v14.4h
    fadd v2.4h,v27.4h,v15.4h
    fadd v3.4h,v28.4h,v16.4h

    fmul v1.4h,v1.4h,v0.h[3]  //mid
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid

    //new (r1+r2),(r1-r2)
    fadd v4.4h,v2.4h,v3.4h
    fsub v5.4h,v2.4h,v3.4h

    //(r1-r2)+2*(r3-r4)
    fadd v2.4h,v29.4h,v17.4h
    fadd v3.4h,v30.4h,v18.4h

    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid

    //new (r3+r4),(r3-r4)*2
    fadd v6.4h,v2.4h,v3.4h
    fsub v7.4h,v2.4h,v3.4h
    fmul v7.4h,v7.4h,v0.h[0]

    prfm    pldl1keep, [x0, 0x200]

    //end-mid ==========================
    fadd v2.4h,v4.4h,v6.4h
    mov  v3.8b,v4.8b
    fadd v1.4h,v1.4h,v2.4h     //v1 done r0+(r1+r2)+(r3+r4)
    fmla v3.4h,v6.4h,v0.h[1]   //v3 done    (r1+r2)+4*(r3+r4)
    fadd v2.4h,v5.4h,v7.4h     //v2 done    (r1-r2)+2*(r3-r4)
    fadd v4.4h,v31.4h,v19.4h   //mid-v4  ============================
    fmul v4.4h,v4.4h,v0.h[3]   //mid
    fmla v4.4h,v7.4h,v0.h[1]
    fadd v4.4h,v4.4h,v5.4h     //v4 done    (r1-r2)+4*(r3-r4)*2 + mid_4

    fmul v1.4h,v1.4h,v0.h[4]  //mul*32
    fmul v3.4h,v3.4h,v0.h[4]
    fmul v2.4h,v2.4h,v0.h[4]
    fmul v4.4h,v4.4h,v0.h[4]
    cbz     x4, activation_1

    add_bias_1:
        ld1r {v6.4h},[x4]
        fadd  v1.4h,v1.4h,v6.4h          //v1+bias
        fadd  v2.4h,v2.4h,v6.4h          //v2+bias
        fadd  v3.4h,v3.4h,v6.4h          //v3+bias
        fadd  v4.4h,v4.4h,v6.4h          //v4+bias
        b activation_1

    activation_1:
    cmp     w5,0
    blt     store_1

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4h, v1.4h, v5.4h
    fmax    v2.4h, v2.4h, v5.4h
    fmax    v3.4h, v3.4h, v5.4h
    fmax    v4.4h, v4.4h, v5.4h

    beq     store_1
    dup     v6.4h,v6.h[0]

    fmin    v1.4h, v1.4h, v6.4h
    fmin    v2.4h, v2.4h, v6.4h
    fmin    v3.4h, v3.4h, v6.4h
    fmin    v4.4h, v4.4h, v6.4h

    store_1:
    st4  {v1.4h,v2.4h,v3.4h,v4.4h}, [x11]

//line2: mid[v1,v4,v5,v6,v7,v(4)]
line2:
    //v1
    mov v1.8b,v20.8b
    mov v2.8b,v21.8b
    mov v3.8b,v22.8b
    fmla v1.4h,v8.4h,v0.h[1]
    fmla v2.4h,v9.4h,v0.h[1]
    fmla v3.4h,v10.4h,v0.h[1]
    fmul v1.4h,v1.4h,v0.h[3]  //mid
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid

    fadd v4.4h,v2.4h,v3.4h
    fsub v5.4h,v2.4h,v3.4h
    mov v2.8b,v23.8b
    mov v3.8b,v24.8b
    fmla v2.4h,v11.4h,v0.h[1]
    fmla v3.4h,v12.4h,v0.h[1]
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid

    fadd v6.4h,v2.4h,v3.4h
    fsub v7.4h,v2.4h,v3.4h
    fmul v7.4h,v7.4h,v0.h[0]
    prfm    pldl1keep, [x0, 0x240]
    //end-mid ==========================
    fadd v2.4h,v4.4h,v6.4h
    mov  v3.8b,v4.8b
    fadd v1.4h,v1.4h,v2.4h     //v1 done r0+(r1+r2)+(r3+r4)
    fmla v3.4h,v6.4h,v0.h[1]   //v3 done    (r1+r2)+4*(r3+r4)
    fadd v2.4h,v5.4h,v7.4h     //v2 done    (r1-r2)+2*(r3-r4)
    mov v4.8b,v25.8b
    fmla v4.4h,v13.4h,v0.h[1]  //mid-v4    //================
    fmul v4.4h,v4.4h,v0.h[3]  //mid
    fmla v4.4h,v7.4h,v0.h[1]
    fadd v4.4h,v4.4h,v5.4h     //v4 done    (r1-r2)+4*(r3-r4)*2 + mid_4

    fmul v1.4h,v1.4h,v0.h[4]  //mul*32
    fmul v3.4h,v3.4h,v0.h[4]
    fmul v2.4h,v2.4h,v0.h[4]
    fmul v4.4h,v4.4h,v0.h[4]
    cbz     x4, activation_2

    add_bias_2:
        ld1r {v6.4h},[x4]
        fadd  v1.4h,v1.4h,v6.4h          //v1+bias
        fadd  v2.4h,v2.4h,v6.4h         //v2+bias
        fadd  v3.4h,v3.4h,v6.4h          //v3+bias
        fadd  v4.4h,v4.4h,v6.4h         //v4+bias
        b activation_2

    activation_2:
    cmp     w5,0
    blt     store_2

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4h, v1.4h, v5.4h
    fmax    v2.4h, v2.4h, v5.4h
    fmax    v3.4h, v3.4h, v5.4h
    fmax    v4.4h, v4.4h, v5.4h

    beq     store_2
    dup     v6.4h,v6.h[0]

    fmin    v1.4h, v1.4h, v6.4h
    fmin    v2.4h, v2.4h, v6.4h
    fmin    v3.4h, v3.4h, v6.4h
    fmin    v4.4h, v4.4h, v6.4h

    store_2:
    st4  {v1.4h,v2.4h,v3.4h,v4.4h}, [x12]


//line0:
line0:
    // add 4 line,free(v8-v13) (r1+r2+r3+r4)
    fadd v20.4h,v20.4h,v8.4h
    fadd v21.4h,v21.4h,v9.4h
    fadd v22.4h,v22.4h,v10.4h
    fadd v23.4h,v23.4h,v11.4h
    fadd v24.4h,v24.4h,v12.4h
    fadd v25.4h,v25.4h,v13.4h
    //load v8-v13
    ldr d8, [x0]
    ldr d9, [x0,#0x8]
    ldr d10, [x0,#0x10]
    ldr d11, [x0,#0x18]
    ldr d12, [x0,#0x20]
    ldr d13, [x0,#0x28]

    //add get mid
    fadd v1.4h,v20.4h,v8.4h
    fadd v2.4h,v21.4h,v9.4h
    fadd v3.4h,v22.4h,v10.4h
    fmul v1.4h,v1.4h,v0.h[3]  //mid
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid

    fadd v4.4h,v2.4h,v3.4h
    fsub v5.4h,v2.4h,v3.4h
    prfm    pldl1keep, [x0, 0x280]
    fadd v2.4h,v23.4h,v11.4h
    fadd v3.4h,v24.4h,v12.4h
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid
    fadd v6.4h,v2.4h,v3.4h
    fsub v7.4h,v2.4h,v3.4h
    fmul v7.4h,v7.4h,v0.h[0]
    //end-mid ==========================
    fadd v2.4h,v4.4h,v6.4h
    mov  v3.8b,v4.8b
    fadd v1.4h,v1.4h,v2.4h     //v1 done r0+(r1+r2)+(r3+r4)
    fmla v3.4h,v6.4h,v0.h[1]   //v3 done    (r1+r2)+4*(r3+r4)
    fadd v2.4h,v5.4h,v7.4h     //v2 done    (r1-r2)+2*(r3-r4)
    fadd v4.4h,v25.4h,v13.4h   //mid-v4
    fmul v4.4h,v4.4h,v0.h[3]  //mid
    fmla v4.4h,v7.4h,v0.h[1]
    fadd v4.4h,v4.4h,v5.4h     //v4 done    (r1-r2)+4*(r3-r4)*2 + mid_4

    fmul v1.4h,v1.4h,v0.h[4]  //mul*32
    fmul v3.4h,v3.4h,v0.h[4]
    fmul v2.4h,v2.4h,v0.h[4]
    fmul v4.4h,v4.4h,v0.h[4]
    cbz     x4, activation_3

    add_bias_3:
        ld1r {v6.4h},[x4]
        fadd  v1.4h,v1.4h,v6.4h          //v1+bias
        fadd  v2.4h,v2.4h,v6.4h         //v2+bias
        fadd  v3.4h,v3.4h,v6.4h          //v3+bias
        fadd  v4.4h,v4.4h,v6.4h         //v4+bias
        b activation_3

    
    activation_3:
    cmp     w5,0
    blt     store_3

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4h, v1.4h, v5.4h
    fmax    v2.4h, v2.4h, v5.4h
    fmax    v3.4h, v3.4h, v5.4h
    fmax    v4.4h, v4.4h, v5.4h

    beq     store_3
    dup     v6.4h,v6.h[0]

    fmin    v1.4h, v1.4h, v6.4h
    fmin    v2.4h, v2.4h, v6.4h
    fmin    v3.4h, v3.4h, v6.4h
    fmin    v4.4h, v4.4h, v6.4h

    store_3:
    st4  {v1.4h,v2.4h,v3.4h,v4.4h}, [x1]

    
//line3:
line3:
    //load v8-v13
    ldr d8,  [x10]
    ldr d9,  [x10,#0x8]
    ldr d10, [x10,#0x10]
    ldr d11, [x10,#0x18]
    ldr d12, [x10,#0x20]
    ldr d13, [x10,#0x28]
    //v1
    fadd v1.4h,v8.4h,v26.4h
    fadd v2.4h,v9.4h,v27.4h
    fadd v3.4h,v10.4h,v28.4h
    fmla v1.4h,v14.4h,v0.h[1]
    fmla v2.4h,v15.4h,v0.h[1]
    fmla v3.4h,v16.4h,v0.h[1]
    fmul v1.4h,v1.4h,v0.h[3]  //mid
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid
    fadd v4.4h,v2.4h,v3.4h
    fsub v5.4h,v2.4h,v3.4h

    fadd v2.4h,v11.4h,v29.4h
    fadd v3.4h,v12.4h,v30.4h
    fmla v2.4h,v17.4h,v0.h[1]
    fmla v3.4h,v18.4h,v0.h[1]
    fmul v2.4h,v2.4h,v0.h[3]  //mid
    fmul v3.4h,v3.4h,v0.h[3]  //mid
    prfm    pldl1keep, [x0, 0x2c0]
    fadd v6.4h,v2.4h,v3.4h
    fsub v7.4h,v2.4h,v3.4h
    fmul v7.4h,v7.4h,v0.h[0]
    //end-mid ==========================
    fadd v2.4h,v4.4h,v6.4h
    mov  v3.8b,v4.8b
    fadd v1.4h,v1.4h,v2.4h     //v1 done r0+(r1+r2)+(r3+r4)
    fmla v3.4h,v6.4h,v0.h[1]   //v3 done    (r1+r2)+4*(r3+r4)
    fadd v2.4h,v5.4h,v7.4h     //v2 done    (r1-r2)+2*(r3-r4)
    fadd v4.4h,v13.4h,v31.4h   
    fmla v4.4h,v19.4h,v0.h[1] //mid-v4
    fmul v4.4h,v4.4h,v0.h[3]  //mid
    fmla v4.4h,v7.4h,v0.h[1]
    fadd v4.4h,v4.4h,v5.4h     //v4 done    (r1-r2)+4*(r3-r4)*2 + mid_4
    fmul v1.4h,v1.4h,v0.h[4]  //mul*32
    fmul v3.4h,v3.4h,v0.h[4]
    fmul v2.4h,v2.4h,v0.h[4]
    fmul v4.4h,v4.4h,v0.h[4]
    cbz     x4, activation_4

    add_bias_4:
        ld1r {v6.4h},[x4]
        fadd  v1.4h,v1.4h,v6.4h          //v1+bias
        fadd  v2.4h,v2.4h,v6.4h         //v2+bias
        fadd  v3.4h,v3.4h,v6.4h          //v3+bias
        fadd  v4.4h,v4.4h,v6.4h         //v4+bias
        b activation_4
    
    activation_4:
    cmp     w5,0
    blt     store_4

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4h, v1.4h, v5.4h
    fmax    v2.4h, v2.4h, v5.4h
    fmax    v3.4h, v3.4h, v5.4h
    fmax    v4.4h, v4.4h, v5.4h

    beq     store_4
    dup     v6.4h,v6.h[0]

    fmin    v1.4h, v1.4h, v6.4h
    fmin    v2.4h, v2.4h, v6.4h
    fmin    v3.4h, v3.4h, v6.4h
    fmin    v4.4h, v4.4h, v6.4h

    store_4:
    st4  {v1.4h,v2.4h,v3.4h,v4.4h}, [x13]


return:
    ldp    d8,  d9,  [sp]
    ldp    d10, d11, [sp, 0x10]
    ldp    d12, d13, [sp, 0x20]
    ldp    d14, d15, [sp, 0x30]
    add    sp, sp, 0x40
    ret
        .end

