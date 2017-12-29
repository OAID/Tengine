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

//scale_neon_bias
.text
.align 5
.global scale_neon_bias
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

    lsr x9,x3,5
    lsl x10,x9,5
    sub x10,x3,x10
    cbz x9, bias_less_32
    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    sub x9,x9,1
    cbz x9, bias_last_block_32

bias_block_32_start:

    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    fmul v8.4s,v0.4s,v30.s[0]
    fadd  v8.4s,v8.4s,v29.4s

    fmul v9.4s,v1.4s,v30.s[0]
    fadd  v9.4s,v9.4s,v29.4s

    fmul v10.4s,v2.4s,v30.s[0]
    fadd  v10.4s,v10.4s,v29.4s

    fmul v11.4s,v3.4s,v30.s[0]
    fadd  v11.4s,v11.4s,v29.4s

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    subs x9,x9,1

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64

    fmul v12.4s,v4.4s,v30.s[0]
    fadd  v12.4s,v12.4s,v29.4s

    fmul v13.4s,v5.4s,v30.s[0]
    fadd  v13.4s,v13.4s,v29.4s

    fmul v14.4s,v6.4s,v30.s[0]
    fadd  v14.4s,v14.4s,v29.4s

    fmul v15.4s,v7.4s,v30.s[0]
    fadd  v15.4s,v15.4s,v29.4s

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64

    b.ne bias_block_32_start

bias_last_block_32:

    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

    fmul v8.4s,v0.4s,v30.s[0]
    fadd  v8.4s,v8.4s,v29.4s

    fmul v9.4s,v1.4s,v30.s[0]
    fadd  v9.4s,v9.4s,v29.4s

    fmul v10.4s,v2.4s,v30.s[0]
    fadd  v10.4s,v10.4s,v29.4s

    fmul v11.4s,v3.4s,v30.s[0]
    fadd  v11.4s,v11.4s,v29.4s

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x8],#64

    fmul v12.4s,v4.4s,v30.s[0]
    fadd  v12.4s,v12.4s,v29.4s

    fmul v13.4s,v5.4s,v30.s[0]
    fadd  v13.4s,v13.4s,v29.4s

    fmul v14.4s,v6.4s,v30.s[0]
    fadd  v14.4s,v14.4s,v29.4s

    fmul v15.4s,v7.4s,v30.s[0]
    fadd  v15.4s,v15.4s,v29.4s

    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x8],#64
    cbz  x10, bias_channel_done

bias_less_32:
    subs x10,x10,1
    ldr s0,[x0],#4
    fmadd s1,s0,s30,s29
    str s1,[x8],#4
    b.ne bias_less_32

bias_channel_done:

    subs x2,x2,1  //channel_counter
    b.ne bias_channel_start

    ret
