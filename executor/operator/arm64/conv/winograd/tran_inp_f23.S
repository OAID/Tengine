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


//x0: inp
//x1: out
//x2: inw
    .section .text,"ax"
    .align 5

    .type tran_inp_f23 STT_FUNC
    .global tran_inp_f23
    .hidden tran_inp_f23
    
tran_inp_f23:

    
comput_idx:
    lsl	x2, x2, 0x2 
    
    add x11,x0,x2      
    add x12,x0,x2,LSL 1  
    add	x13,x11,x2, LSL 1  
    
    add x4,x0,#0x8
    add x5,x11,#0x8 
    add x6,x12,#0x8   
    add x7,x13,#0x8


    ld2 {v0.4s, v1.4s},[x0]
    ld2 {v2.4s, v3.4s},[x4]
    ld2 {v4.4s, v5.4s},[x12]
    ld2 {v6.4s, v7.4s},[x6]
   
    fsub	v18.4s, v0.4s, v4.4s
    fsub	v19.4s, v1.4s, v5.4s
    fsub	v20.4s, v2.4s, v6.4s
    fsub	v21.4s, v3.4s, v7.4s
  
    fsub    v22.4s, v18.4s,v20.4s
    fadd    v23.4s, v19.4s,v20.4s
    fsub    v24.4s, v20.4s,v19.4s
    fsub    v25.4s, v21.4s,v19.4s
    stp q22,q23,[x1]
    stp q24,q25,[x1,0x20]

    ld2 {v0.4s, v1.4s},[x11]
    ld2 {v2.4s, v3.4s},[x5]
    fadd	v18.4s, v0.4s, v4.4s
    fadd	v19.4s, v1.4s, v5.4s
    fadd	v20.4s, v2.4s, v6.4s
    fadd	v21.4s, v3.4s, v7.4s
  
    fsub    v22.4s, v18.4s,v20.4s
    fadd    v23.4s, v19.4s,v20.4s
    fsub    v24.4s, v20.4s,v19.4s
    fsub    v25.4s, v21.4s,v19.4s
    stp	 q22,q23, [x1, 0x40]
    stp	 q24,q25, [x1, 0x60]

    fsub	v18.4s, v4.4s, v0.4s
    fsub	v19.4s, v5.4s, v1.4s
    fsub	v20.4s, v6.4s, v2.4s
    fsub	v21.4s, v7.4s, v3.4s

    fsub    v22.4s, v18.4s,v20.4s
    fadd    v23.4s, v19.4s,v20.4s
    fsub    v24.4s, v20.4s,v19.4s
    fsub    v25.4s, v21.4s,v19.4s
    stp	 q22,q23, [x1, 0x80]
    stp	 q24,q25, [x1, 0xa0]

    ld2 {v4.4s, v5.4s},[x13]
    ld2 {v6.4s, v7.4s},[x7]
    fsub	v18.4s, v4.4s, v0.4s
    fsub	v19.4s, v5.4s, v1.4s
    fsub	v20.4s, v6.4s, v2.4s
    fsub	v21.4s, v7.4s, v3.4s
  
    fsub    v22.4s, v18.4s,v20.4s
    fadd    v23.4s, v19.4s,v20.4s
    fsub    v24.4s, v20.4s,v19.4s
    fsub    v25.4s, v21.4s,v19.4s
    
    stp	 q22,q23, [x1, 0xc0]
    stp	 q24,q25, [x1, 0xe0]

return:

	ret
        .end

