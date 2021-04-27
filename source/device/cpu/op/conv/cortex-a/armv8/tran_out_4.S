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
 * Author: chunyinglv@openailab.com
*/


//x0: buffer
//x1: output
//x2: out_hw*sizeof(float)
//x3: ker
//x4: bias
//x5: activation

    .section .text,"ax"
    .align 5

    .type tran_out_4 STT_FUNC
    .global tran_out_4
    .hidden tran_out_4
    
tran_out_4:
    sub	sp, sp, 0x40
    stp	d8, d9, [sp]
	stp	d10,d11,[sp, 0x10]
	stp	d12,d13,[sp, 0x20]
	stp	d14,d15,[sp, 0x30]
    
comput_idx:
    //str[x1,x11,x12,x13]
    add x11,x1,x2      
    add x12,x1,x2,LSL 1     
    add	x13,x11,x2, LSL 1  
    //ldr[x0,x8,x9,x10]
    add x8,x0,#0x60
    add x9,x0,#0x120
    add x10,x0,#0x1e0

load:
    //load v0-v11
    //add:v20-v25
    //sub:v26-v31
    
    ldp q0,q1,[x8]
    ldp q2,q3,[x8,#0x20]
    ldp q4,q5,[x8,#0x40]
    ldp q6,q7,[x8,#0x60]
    ldp q8,q9,[x8,#0x80]
    ldp q10,q11,[x8,#0xa0]

    fadd v20.4s,v0.4s,v6.4s
    fadd v21.4s,v1.4s,v7.4s
    fadd v22.4s,v2.4s,v8.4s
    fadd v23.4s,v3.4s,v9.4s
    fadd v24.4s,v4.4s,v10.4s
    fadd v25.4s,v5.4s,v11.4s
    
    fsub v26.4s,v0.4s,v6.4s
    fsub v27.4s,v1.4s,v7.4s
    fsub v28.4s,v2.4s,v8.4s
    fsub v29.4s,v3.4s,v9.4s
    fsub v30.4s,v4.4s,v10.4s
    fsub v31.4s,v5.4s,v11.4s

    //load:v0-v7
    //add:v8-v13
    //sub:v14-v19
    ldp q0,q1,[x9]
    ldp q2,q3,[x9,#0x20]
    ldp q4,q5,[x9,#0x60]
    ldp q6,q7,[x9,#0x80]

    fadd v8.4s,v0.4s,v4.4s
    fadd v9.4s,v1.4s,v5.4s
    fadd v10.4s,v2.4s,v6.4s
    fadd v11.4s,v3.4s,v7.4s
  
    fsub v14.4s,v0.4s,v4.4s
    fsub v15.4s,v1.4s,v5.4s
    fsub v16.4s,v2.4s,v6.4s
    fsub v17.4s,v3.4s,v7.4s

    ldp q0,q1,[x9,#0x40]
    ldp q2,q3,[x9,#0xa0]
    fadd v12.4s,v0.4s,v2.4s
    fadd v13.4s,v1.4s,v3.4s
    fsub v18.4s,v0.4s,v2.4s
    fsub v19.4s,v1.4s,v3.4s

    ldr q0,[x3]

//line1: mid[v1,v4,v5,v6,v7,v(4)]
line1:
    mov v1.16b,v26.16b
    mov v2.16b,v27.16b
    mov v3.16b,v28.16b
    fmla v1.4s,v14.4s,v0.s[0]
    fmla v2.4s,v15.4s,v0.s[0]
    fmla v3.4s,v16.4s,v0.s[0]
    fadd v4.4s,v2.4s,v3.4s
    fsub v5.4s,v2.4s,v3.4s
    mov v2.16b,v29.16b
    mov v3.16b,v30.16b
    fmla v2.4s,v17.4s,v0.s[0]
    fmla v3.4s,v18.4s,v0.s[0]
    fadd v6.4s,v2.4s,v3.4s
    fsub v7.4s,v2.4s,v3.4s
    prfm	pldl1keep, [x0, 0x200]

    //end-mid ==========================
    fadd v2.4s,v4.4s,v6.4s
    mov  v3.16b,v4.16b
    fadd v1.4s,v1.4s,v2.4s
    fmla v3.4s,v6.4s,v0.s[1]

    cbz     x4, none_biases_1

    //bias
    ld1r {v6.4s},[x4]
    mov v4.16b,v31.16b
    fadd v1.4s,v1.4s,v6.4s          //v1+bias
    fmla v4.4s,v19.4s,v0.s[0]
    fadd  v2.4s,v6.4s,v5.4s         //v2+bias
    fadd  v4.4s,v4.4s,v5.4s 
    fadd v3.4s,v3.4s,v6.4s          //v3+bias
    fadd  v4.4s,v4.4s,v6.4s         //v4+bias
    fmla v2.4s,v7.4s,v0.s[0]
    fmla v4.4s,v7.4s,v0.s[2]
    b activation_1

    none_biases_1:
    mov v4.16b,v31.16b
    mov  v2.16b,v5.16b
    fmla v4.4s,v19.4s,v0.s[0]
    fmla v2.4s,v7.4s,v0.s[0]
    fadd v4.4s,v4.4s,v5.4s
    fmla v4.4s,v7.4s,v0.s[2]

    activation_1:
    cmp     w5,0
    blt     store_1

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4s, v1.4s, v5.4s
    fmax    v2.4s, v2.4s, v5.4s
    fmax    v3.4s, v3.4s, v5.4s
    fmax    v4.4s, v4.4s, v5.4s

    beq     store_1
    dup     v6.4s,v6.s[0]

    fmin    v1.4s, v1.4s, v6.4s
    fmin    v2.4s, v2.4s, v6.4s
    fmin    v3.4s, v3.4s, v6.4s
    fmin    v4.4s, v4.4s, v6.4s

    store_1:
    st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x11]

//line2: mid[v1,v4,v5,v6,v7,v(4)]
line2:
    //v1
    mov v1.16b,v20.16b
    mov v2.16b,v21.16b
    mov v3.16b,v22.16b
    fmla v1.4s,v8.4s,v0.s[1]
    fmla v2.4s,v9.4s,v0.s[1]
    fmla v3.4s,v10.4s,v0.s[1]
    fadd v4.4s,v2.4s,v3.4s
    fsub v5.4s,v2.4s,v3.4s
    mov v2.16b,v23.16b
    mov v3.16b,v24.16b
    fmla v2.4s,v11.4s,v0.s[1]
    fmla v3.4s,v12.4s,v0.s[1]
    fadd v6.4s,v2.4s,v3.4s
    fsub v7.4s,v2.4s,v3.4s
    prfm	pldl1keep, [x0, 0x240]
    //end-mid ==========================
    fadd v2.4s,v4.4s,v6.4s
    mov  v3.16b,v4.16b
    fadd v1.4s,v1.4s,v2.4s
    fmla v3.4s,v6.4s,v0.s[1]
    
    cbz     x4, none_biases_2

    //bias
    ld1r {v6.4s},[x4]
    mov v4.16b,v25.16b
    fadd v1.4s,v1.4s,v6.4s          //v1+bias
    fmla v4.4s,v13.4s,v0.s[1]
    fadd  v2.4s,v6.4s,v5.4s         //v2+bias
    fadd  v4.4s,v4.4s,v5.4s 
    fadd v3.4s,v3.4s,v6.4s          //v3+bias
    fadd  v4.4s,v4.4s,v6.4s         //v4+bias
    fmla v2.4s,v7.4s,v0.s[0]
    fmla v4.4s,v7.4s,v0.s[2]
    b activation_2

    none_biases_2:
    mov v4.16b,v25.16b
    mov  v2.16b,v5.16b
    fmla v4.4s,v13.4s,v0.s[1]
    fmla v2.4s,v7.4s,v0.s[0]
    fadd v4.4s,v4.4s,v5.4s
    fmla v4.4s,v7.4s,v0.s[2]

    
    activation_2:
    cmp     w5,0
    blt     store_2

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4s, v1.4s, v5.4s
    fmax    v2.4s, v2.4s, v5.4s
    fmax    v3.4s, v3.4s, v5.4s
    fmax    v4.4s, v4.4s, v5.4s

    beq     store_2
    dup     v6.4s,v6.s[0]

    fmin    v1.4s, v1.4s, v6.4s
    fmin    v2.4s, v2.4s, v6.4s
    fmin    v3.4s, v3.4s, v6.4s
    fmin    v4.4s, v4.4s, v6.4s

    store_2:
    st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x12]


//line0:
line0:
    // add 4 line,free(v8-v13)
    fadd v20.4s,v20.4s,v8.4s
    fadd v21.4s,v21.4s,v9.4s
    fadd v22.4s,v22.4s,v10.4s
    fadd v23.4s,v23.4s,v11.4s
    fadd v24.4s,v24.4s,v12.4s
    fadd v25.4s,v25.4s,v13.4s
    //load v8-v13
    ldp q8,q9,[x0]
    ldp q10,q11,[x0,#0x20]
    ldp q12,q13,[x0,#0x40]
    //add get mid
    fadd v1.4s,v20.4s,v8.4s
    fadd v2.4s,v21.4s,v9.4s
    fadd v3.4s,v22.4s,v10.4s
    fadd v4.4s,v2.4s,v3.4s
    fsub v5.4s,v2.4s,v3.4s
    prfm	pldl1keep, [x0, 0x280]
    fadd v2.4s,v23.4s,v11.4s
    fadd v3.4s,v24.4s,v12.4s
    fadd v6.4s,v2.4s,v3.4s
    fsub v7.4s,v2.4s,v3.4s
    //end-mid ==========================
    fadd v2.4s,v4.4s,v6.4s
    mov  v3.16b,v4.16b
    fadd v1.4s,v1.4s,v2.4s
    fmla v3.4s,v6.4s,v0.s[1]
    
    cbz     x4, none_biases_3
    //bias
    ld1r {v6.4s},[x4]
    fadd v4.4s,v25.4s,v13.4s
    fadd v1.4s,v1.4s,v6.4s //v1+bias
    fadd v4.4s,v4.4s,v5.4s
    fadd v2.4s,v6.4s,v5.4s//v2+bias
    fmla v4.4s,v7.4s,v0.s[2]
    fadd v3.4s,v3.4s,v6.4s//v3+bias
    fmla v2.4s,v7.4s,v0.s[0]
    fadd v4.4s,v4.4s,v6.4s //v4+bias
    b activation_3

    none_biases_3:
    fadd v4.4s,v25.4s,v13.4s
    mov  v2.16b,v5.16b
    fadd v4.4s,v4.4s,v5.4s
    fmla v2.4s,v7.4s,v0.s[0]
    fmla v4.4s,v7.4s,v0.s[2]

    
    activation_3:
    cmp     w5,0
    blt     store_3

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4s, v1.4s, v5.4s
    fmax    v2.4s, v2.4s, v5.4s
    fmax    v3.4s, v3.4s, v5.4s
    fmax    v4.4s, v4.4s, v5.4s

    beq     store_3
    dup     v6.4s,v6.s[0]

    fmin    v1.4s, v1.4s, v6.4s
    fmin    v2.4s, v2.4s, v6.4s
    fmin    v3.4s, v3.4s, v6.4s
    fmin    v4.4s, v4.4s, v6.4s

    store_3:
    st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x1]

    
//line3:
line3:
    //load v8-v13
    ldp q8,q9,[x10]
    ldp q10,q11,[x10,#0x20]
    ldp q12,q13,[x10,#0x40]
    //v1
    fadd v1.4s,v8.4s,v26.4s
    fadd v2.4s,v9.4s,v27.4s
    fadd v3.4s,v10.4s,v28.4s
    fmla v1.4s,v14.4s,v0.s[2]
    fmla v2.4s,v15.4s,v0.s[2]
    fmla v3.4s,v16.4s,v0.s[2]
    fadd v4.4s,v2.4s,v3.4s
    fsub v5.4s,v2.4s,v3.4s
    fadd v2.4s,v11.4s,v29.4s
    fadd v3.4s,v12.4s,v30.4s
    fmla v2.4s,v17.4s,v0.s[2]
    fmla v3.4s,v18.4s,v0.s[2]
    prfm	pldl1keep, [x0, 0x2c0]
    fadd v6.4s,v2.4s,v3.4s
    fsub v7.4s,v2.4s,v3.4s
    //end-mid ==========================
    fadd v2.4s,v4.4s,v6.4s
    mov  v3.16b,v4.16b
    fadd v1.4s,v1.4s,v2.4s
    fmla v3.4s,v6.4s,v0.s[1]

    cbz     x4, none_biases_4
    //bias
    ld1r {v6.4s},[x4]
    fadd v4.4s,v13.4s,v31.4s
    fadd v1.4s,v1.4s,v6.4s      //v1+bias
    fmla v4.4s,v19.4s,v0.s[2]
    fadd  v2.4s,v6.4s,v5.4s     //v2+bias
    fmla v4.4s,v7.4s,v0.s[2]
    fadd v3.4s,v3.4s,v6.4s      //v3+bias
    fmla v2.4s,v7.4s,v0.s[0]
    fadd v6.4s,v6.4s,v5.4s 
    fadd v4.4s,v4.4s,v6.4s      //v4+bias
    b activation_4

    none_biases_4:
    fadd v4.4s,v13.4s,v31.4s
    mov  v2.16b,v5.16b
    fmla v4.4s,v19.4s,v0.s[2]
    fmla v2.4s,v7.4s,v0.s[0]
    fadd v4.4s,v4.4s,v5.4s
    fmla v4.4s,v7.4s,v0.s[2]
    
    activation_4:
    cmp     w5,0
    blt     store_4

    movi    d5, 0
    scvtf   s6,w5

    fmax    v1.4s, v1.4s, v5.4s
    fmax    v2.4s, v2.4s, v5.4s
    fmax    v3.4s, v3.4s, v5.4s
    fmax    v4.4s, v4.4s, v5.4s

    beq     store_4
    dup     v6.4s,v6.s[0]

    fmin    v1.4s, v1.4s, v6.4s
    fmin    v2.4s, v2.4s, v6.4s
    fmin    v3.4s, v3.4s, v6.4s
    fmin    v4.4s, v4.4s, v6.4s

    store_4:
    st4  {v1.4s,v2.4s,v3.4s,v4.4s}, [x13]


return:
	ldp	d8,  d9,  [sp]
	ldp	d10, d11, [sp, 0x10]
	ldp	d12, d13, [sp, 0x20]
	ldp	d14, d15, [sp, 0x30]
	add	sp, sp, 0x40
	ret
        .end

