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
 * Author: haitao@openailab.com
 */
/* relu implementation using neon vector */


.text
.align 5
.global scale_neon
.hidden scale_neon
.type   scale_neon, %function

scale_neon:

   //x0 input 
   //x1 gamma
   //x2 channel_number
   //x3 channel_size
   //x4 output

    mov x8,x4   //x8, output ptr

channel_start:
    ldr  s30,[x1],#4

    lsr x9,x3,6   //64 elem 
    lsl x10,x9,6
    sub x10,x3,x10
    cbz x9, less_64
    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
    sub x9,x9,1
    cbz x9, last_block_64

block_64_start:
   
    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64
    fmul v8.4s,v0.4s,v30.s[0]
    fmul v9.4s,v1.4s,v30.s[0]
    fmul v10.4s,v2.4s,v30.s[0]
    fmul v11.4s,v3.4s,v30.s[0]
    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64
    fmul v12.4s,v4.4s,v30.s[0]
    fmul v13.4s,v5.4s,v30.s[0]
    fmul v14.4s,v6.4s,v30.s[0]
    fmul v15.4s,v7.4s,v30.s[0]
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    subs x9,x9,1

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    fmul v8.4s,v16.4s,v30.s[0]
    fmul v9.4s,v17.4s,v30.s[0]
    fmul v10.4s,v18.4s,v30.s[0]
    fmul v11.4s,v19.4s,v30.s[0]
    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
    fmul v12.4s,v20.4s,v30.s[0]
    fmul v13.4s,v21.4s,v30.s[0]
    fmul v14.4s,v22.4s,v30.s[0]
    fmul v15.4s,v23.4s,v30.s[0]
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64


    b.ne block_64_start

last_block_64:

    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64
    fmul v8.4s,v0.4s,v30.s[0]
    fmul v9.4s,v1.4s,v30.s[0]
    fmul v10.4s,v2.4s,v30.s[0]
    fmul v11.4s,v3.4s,v30.s[0]
    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64
    fmul v12.4s,v4.4s,v30.s[0]
    fmul v13.4s,v5.4s,v30.s[0]
    fmul v14.4s,v6.4s,v30.s[0]
    fmul v15.4s,v7.4s,v30.s[0]
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    fmul v8.4s,v16.4s,v30.s[0]
    fmul v9.4s,v17.4s,v30.s[0]
    fmul v10.4s,v18.4s,v30.s[0]
    fmul v11.4s,v19.4s,v30.s[0]
    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    fmul v12.4s,v20.4s,v30.s[0]
    fmul v13.4s,v21.4s,v30.s[0]
    fmul v14.4s,v22.4s,v30.s[0]
    fmul v15.4s,v23.4s,v30.s[0]
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    cbz  x10, channel_done

less_64:
    subs x10,x10,1
    ldr s0,[x0],#4
    fmul s1,s0,s30
    str s1,[x8],#4
    b.ne less_64

channel_done:

    subs x2,x2,1  //channel_counter
    b.ne channel_start

    ret


//scale_neon_bias
.global scale_neon_bias
.hidden scale_neon_bias
.type   scale_neon_bias, %function

scale_neon_bias:
   //x0 input 
   //x1 gamma
   //x2 channel_number
   //x3 channel_size
   //x4 output
   //x5 bias 

    mov x8,x4   //x8, output ptr

bias_channel_start:
    ldr  s30,[x1],#4
    ld1r {v29.4s},[x5],#4 

    lsr x9,x3,6
    lsl x10,x9,6
    sub x10,x3,x10
    cbz x9, bias_less_64

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    sub x9,x9,1
    cbz x9, bias_last_block_64

bias_block_64_start:

    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64

    mov v8.16b,v29.16b
    fmla v8.4s,v0.4s,v30.s[0]

    mov v9.16b,v29.16b
    fmla v9.4s,v1.4s,v30.s[0]

    mov v10.16b,v29.16b
    fmla v10.4s,v2.4s,v30.s[0]

    mov v11.16b,v29.16b
    fmla v11.4s,v3.4s,v30.s[0]

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64

    mov v12.16b,v29.16b
    fmla v12.4s,v4.4s,v30.s[0]

    mov v13.16b,v29.16b
    fmla v13.4s,v5.4s,v30.s[0]

    mov v14.16b,v29.16b
    fmla v14.4s,v6.4s,v30.s[0]

    mov v15.16b,v29.16b
    fmla v15.4s,v7.4s,v30.s[0]

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    subs x9,x9,1

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64

    mov v8.16b,v29.16b
    fmla v8.4s,v16.4s,v30.s[0]

    mov v9.16b,v29.16b
    fmla v9.4s,v17.4s,v30.s[0]

    mov v10.16b,v29.16b
    fmla v10.4s,v18.4s,v30.s[0]

    mov v11.16b,v29.16b
    fmla v11.4s,v19.4s,v30.s[0]

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    mov v12.16b,v29.16b
    fmla v12.4s,v20.4s,v30.s[0]

    mov v13.16b,v29.16b
    fmla v13.4s,v21.4s,v30.s[0]

    mov v14.16b,v29.16b
    fmla v14.4s,v22.4s,v30.s[0]

    mov v15.16b,v29.16b
    fmla v15.4s,v23.4s,v30.s[0]

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    b.ne bias_block_64_start

bias_last_block_64:

    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x0],#64

    mov v8.16b,v29.16b
    fmla v8.4s,v0.4s,v30.s[0]

    mov v9.16b,v29.16b
    fmla v9.4s,v1.4s,v30.s[0]

    mov v10.16b,v29.16b
    fmla v10.4s,v2.4s,v30.s[0]

    mov v11.16b,v29.16b
    fmla v11.4s,v3.4s,v30.s[0]

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x0],#64

    mov v12.16b,v29.16b
    fmla v12.4s,v4.4s,v30.s[0]

    mov v13.16b,v29.16b
    fmla v13.4s,v5.4s,v30.s[0]

    mov v14.16b,v29.16b
    fmla v14.4s,v6.4s,v30.s[0]

    mov v15.16b,v29.16b
    fmla v15.4s,v7.4s,v30.s[0]
    
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    mov v8.16b,v29.16b
    fmla v8.4s,v16.4s,v30.s[0]

    mov v9.16b,v29.16b
    fmla v9.4s,v17.4s,v30.s[0]

    mov v10.16b,v29.16b
    fmla v10.4s,v18.4s,v30.s[0]

    mov v11.16b,v29.16b
    fmla v11.4s,v19.4s,v30.s[0]

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    mov v12.16b,v29.16b
    fmla v12.4s,v20.4s,v30.s[0]

    mov v13.16b,v29.16b
    fmla v13.4s,v21.4s,v30.s[0]

    mov v14.16b,v29.16b
    fmla v14.4s,v22.4s,v30.s[0]

    mov v15.16b,v29.16b
    fmla v15.4s,v23.4s,v30.s[0]
    
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64
    
    cbz  x10, bias_channel_done

bias_less_64:
    subs x10,x10,1
    ldr s0,[x0],#4
    fmadd s1,s0,s30,s29
    str s1,[x8],#4
    b.ne bias_less_64

bias_channel_done:

    subs x2,x2,1  //channel_counter
    b.ne bias_channel_start

    ret
