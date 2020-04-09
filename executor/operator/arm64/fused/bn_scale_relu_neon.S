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
 * Author: haitao@openailab.com
 */
/* relu implementation using neon vector */


.text
.align 5
.global bn_scale_relu_neon
.hidden bn_scale_relu_neon
.type   bn_scale_relu_neon, %function

bn_scale_relu_neon:
   //x0 input 
   //x1 gamma
   //x2 beta
   //x3 mean
   //x4 var
   //x5 channel_number
   //x6 channel_size
   //x7 output

   //s28 -- gamma
   //s29 -- beta
   //s30 -- mean
   //s31 -- var
   //v27 --- zero

   /* 
     data=data*s_var+s_mean;
     data=data*s_gamma+s_beta;
   */

    fmov s27,wzr
    dup  v27.4s,v27.s[0]

channel_start:

    ldr  s28,[x1],#4
    ld1r {v29.4s},[x2],#4 
    ld1r {v30.4s},[x3],#4
    ldr  s31,[x4],#4

    lsr x9,x6,6
    lsl x10,x9,6
    sub x10,x6,x10
    cbz x9, less_64

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    sub x9,x9,1
    cbz x9, last_block_64

block_64_start:


    mov v8.16b,v30.16b
    mov v9.16b,v30.16b
    mov v10.16b,v30.16b
    mov v11.16b,v30.16b

    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64

    fmla v8.4s,v0.4s,v31.s[0]
    fmla v9.4s,v1.4s,v31.s[0]
    fmla v10.4s,v2.4s,v31.s[0]
    fmla v11.4s,v3.4s,v31.s[0]


    fmul v8.4s,v8.4s,v28.s[0]
    fmul v9.4s,v9.4s,v28.s[0]
    fmul v10.4s,v10.4s,v28.s[0]
    fmul v11.4s,v11.4s,v28.s[0]

    fadd v8.4s,v8.4s,v29.4s
    fadd v9.4s,v9.4s,v29.4s
    fadd v10.4s,v10.4s,v29.4s
    fadd v11.4s,v11.4s,v29.4s

    fmax v8.4s,v8.4s,v27.4s
    fmax v9.4s,v9.4s,v27.4s
    fmax v10.4s,v10.4s,v27.4s
    fmax v11.4s,v11.4s,v27.4s

    mov v12.16b,v30.16b
    mov v13.16b,v30.16b

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x7],#64

    mov v14.16b,v30.16b
    mov v15.16b,v30.16b

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64

    fmla v12.4s,v4.4s,v31.s[0]
    fmla v13.4s,v5.4s,v31.s[0]
    fmla v14.4s,v6.4s,v31.s[0]
    fmla v15.4s,v7.4s,v31.s[0]

    fmul v12.4s,v12.4s,v28.s[0]
    fmul v13.4s,v13.4s,v28.s[0]
    fmul v14.4s,v14.4s,v28.s[0]
    fmul v15.4s,v15.4s,v28.s[0]

    fadd v12.4s,v12.4s,v29.4s
    fadd v13.4s,v13.4s,v29.4s
    fadd v14.4s,v14.4s,v29.4s
    fadd v15.4s,v15.4s,v29.4s

    fmax v12.4s,v12.4s,v27.4s
    fmax v13.4s,v13.4s,v27.4s
    fmax v14.4s,v14.4s,v27.4s
    fmax v15.4s,v15.4s,v27.4s

    mov v8.16b,v30.16b
    mov v9.16b,v30.16b

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x7],#64

    mov v10.16b,v30.16b
    mov v11.16b,v30.16b
    subs x9,x9,1

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64

    fmla v8.4s,v16.4s,v31.s[0]
    fmla v9.4s,v17.4s,v31.s[0]
    fmla v10.4s,v18.4s,v31.s[0]
    fmla v11.4s,v19.4s,v31.s[0]

    fmul v8.4s,v8.4s,v28.s[0]
    fmul v9.4s,v9.4s,v28.s[0]
    fmul v10.4s,v10.4s,v28.s[0]
    fmul v11.4s,v11.4s,v28.s[0]

    fadd v8.4s,v8.4s,v29.4s
    fadd v9.4s,v9.4s,v29.4s
    fadd v10.4s,v10.4s,v29.4s
    fadd v11.4s,v11.4s,v29.4s

    fmax v8.4s,v8.4s,v27.4s
    fmax v9.4s,v9.4s,v27.4s
    fmax v10.4s,v10.4s,v27.4s
    fmax v11.4s,v11.4s,v27.4s


    mov v12.16b,v30.16b
    mov v13.16b,v30.16b

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x7],#64

    mov v14.16b,v30.16b
    mov v15.16b,v30.16b

    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    fmla v12.4s,v20.4s,v31.s[0]
    fmla v13.4s,v21.4s,v31.s[0]
    fmla v14.4s,v22.4s,v31.s[0]
    fmla v15.4s,v23.4s,v31.s[0]

    fmul v12.4s,v12.4s,v28.s[0]
    fmul v13.4s,v13.4s,v28.s[0]
    fmul v14.4s,v14.4s,v28.s[0]
    fmul v15.4s,v15.4s,v28.s[0]

    fadd v12.4s,v12.4s,v29.4s
    fadd v13.4s,v13.4s,v29.4s
    fadd v14.4s,v14.4s,v29.4s
    fadd v15.4s,v15.4s,v29.4s

    fmax v12.4s,v12.4s,v27.4s
    fmax v13.4s,v13.4s,v27.4s
    fmax v14.4s,v14.4s,v27.4s
    fmax v15.4s,v15.4s,v27.4s

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x7],#64

    b.ne block_64_start

last_block_64:

    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64

    mov v8.16b,v30.16b
    mov v9.16b,v30.16b
    mov v10.16b,v30.16b
    mov v11.16b,v30.16b

    fmla v8.4s,v0.4s,v31.s[0]
    fmla v9.4s,v1.4s,v31.s[0]
    fmla v10.4s,v2.4s,v31.s[0]
    fmla v11.4s,v3.4s,v31.s[0]

    fmul v8.4s,v8.4s,v28.s[0]
    fmul v9.4s,v9.4s,v28.s[0]
    fmul v10.4s,v10.4s,v28.s[0]
    fmul v11.4s,v11.4s,v28.s[0]

    fadd v8.4s,v8.4s,v29.4s
    fadd v9.4s,v9.4s,v29.4s
    fadd v10.4s,v10.4s,v29.4s
    fadd v11.4s,v11.4s,v29.4s

    fmax v8.4s,v8.4s,v27.4s
    fmax v9.4s,v9.4s,v27.4s
    fmax v10.4s,v10.4s,v27.4s
    fmax v11.4s,v11.4s,v27.4s

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x7],#64

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64

    mov v12.16b,v30.16b
    mov v13.16b,v30.16b
    mov v14.16b,v30.16b
    mov v15.16b,v30.16b

    fmla v12.4s,v4.4s,v31.s[0]
    fmla v13.4s,v5.4s,v31.s[0]
    fmla v14.4s,v6.4s,v31.s[0]
    fmla v15.4s,v7.4s,v31.s[0]

    fmul v12.4s,v12.4s,v28.s[0]
    fmul v13.4s,v13.4s,v28.s[0]
    fmul v14.4s,v14.4s,v28.s[0]
    fmul v15.4s,v15.4s,v28.s[0]

    fadd v12.4s,v12.4s,v29.4s
    fadd v13.4s,v13.4s,v29.4s
    fadd v14.4s,v14.4s,v29.4s
    fadd v15.4s,v15.4s,v29.4s

    fmax v12.4s,v12.4s,v27.4s
    fmax v13.4s,v13.4s,v27.4s
    fmax v14.4s,v14.4s,v27.4s
    fmax v15.4s,v15.4s,v27.4s


    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x7],#64

    mov v8.16b,v30.16b
    mov v9.16b,v30.16b
    mov v10.16b,v30.16b
    mov v11.16b,v30.16b

    fmla v8.4s,v16.4s,v31.s[0]
    fmla v9.4s,v17.4s,v31.s[0]
    fmla v10.4s,v18.4s,v31.s[0]
    fmla v11.4s,v19.4s,v31.s[0]

    fmul v8.4s,v8.4s,v28.s[0]
    fmul v9.4s,v9.4s,v28.s[0]
    fmul v10.4s,v10.4s,v28.s[0]
    fmul v11.4s,v11.4s,v28.s[0]

    fadd v8.4s,v8.4s,v29.4s
    fadd v9.4s,v9.4s,v29.4s
    fadd v10.4s,v10.4s,v29.4s
    fadd v11.4s,v11.4s,v29.4s

    fmax v8.4s,v8.4s,v27.4s
    fmax v9.4s,v9.4s,v27.4s
    fmax v10.4s,v10.4s,v27.4s
    fmax v11.4s,v11.4s,v27.4s

    mov v12.16b,v30.16b
    mov v13.16b,v30.16b
    mov v14.16b,v30.16b
    mov v15.16b,v30.16b

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x7],#64

    fmla v12.4s,v20.4s,v31.s[0]
    fmla v13.4s,v21.4s,v31.s[0]
    fmla v14.4s,v22.4s,v31.s[0]
    fmla v15.4s,v23.4s,v31.s[0]

    fmul v12.4s,v12.4s,v28.s[0]
    fmul v13.4s,v13.4s,v28.s[0]
    fmul v14.4s,v14.4s,v28.s[0]
    fmul v15.4s,v15.4s,v28.s[0]

    fadd v12.4s,v12.4s,v29.4s
    fadd v13.4s,v13.4s,v29.4s
    fadd v14.4s,v14.4s,v29.4s
    fadd v15.4s,v15.4s,v29.4s

    fmax v12.4s,v12.4s,v27.4s
    fmax v13.4s,v13.4s,v27.4s
    fmax v14.4s,v14.4s,v27.4s
    fmax v15.4s,v15.4s,v27.4s

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x7],#64            
    
    cbz  x10, channel_done

less_64:
    subs x10,x10,1
    ldr s0,[x0],#4
    fmadd s1,s0,s31,s30
    fmadd s1,s1,s28,s29
    fmax  s1,s1,s27
    str s1,[x7],#4
    b.ne less_64

channel_done:

    subs x5,x5,1  //channel_counter
    b.ne channel_start

    ret
