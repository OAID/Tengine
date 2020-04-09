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


//x0: mid
//x1: tmp
//x2: tmp1
//x3: bias
//x4: activation

    .section .text,"ax"
    .align 5

    .type tran_out_f23 STT_FUNC
    .global tran_out_f23
    .hidden tran_out_f23
    
tran_out_f23:

cbz     x3, non_bias  
    ld1r   {v16.4s},[x3]

non_bias:

movi    d17, 0

line0:
    ldp q0,q1,[x0]
    ldp q2,q3,[x0,#0x20]
    ldp q4,q5,[x0,#0x40]
    ldp q6,q7,[x0,#0x60]
    ldp q18,q19,[x0,#0x80]
    ldp q20,q21,[x0,#0xa0]

    fadd v22.4s,v0.4s,v4.4s
    fadd v23.4s,v1.4s,v5.4s
    fadd v24.4s,v2.4s,v6.4s
    fadd v25.4s,v3.4s,v7.4s

    fadd v26.4s,v22.4s,v18.4s
    fadd v27.4s,v23.4s,v19.4s
    fadd v28.4s,v24.4s,v20.4s
    fadd v29.4s,v25.4s,v21.4s

    //end-mid

    fadd v30.4s,v26.4s,v27.4s
    fsub v31.4s,v27.4s,v28.4s
    fadd v30.4s,v30.4s,v28.4s
    fadd v31.4s,v31.4s,v29.4s

    //bias?
    cbz     x3, activation_1    
    fadd v30.4s,v30.4s,v16.4s          //v1+bias
    fadd v31.4s,v31.4s,v16.4s          //v2+bias

    activation_1:
    cmp     w4,0
    blt     store_1

    fmax    v30.4s, v30.4s, v17.4s
    fmax    v31.4s, v31.4s, v17.4s
    beq     store_1

    store_1:
    //stp q30,q31,[x1]
    st2  {v30.4s,v31.4s}, [x1]
  
line1:
    ldp q0,q1,[x0,#0xc0]
    ldp q2,q3,[x0,#0xe0]

    fadd v22.4s,v0.4s,v4.4s
    fadd v23.4s,v1.4s,v5.4s
    fadd v24.4s,v2.4s,v6.4s
    fadd v25.4s,v3.4s,v7.4s

    fsub v26.4s,v22.4s,v18.4s
    fsub v27.4s,v23.4s,v19.4s
    fsub v28.4s,v24.4s,v20.4s
    fsub v29.4s,v25.4s,v21.4s

    //end-mid
    fadd v30.4s,v26.4s,v27.4s
    fsub v31.4s,v27.4s,v28.4s
    fadd v30.4s,v30.4s,v28.4s
    fadd v31.4s,v31.4s,v29.4s

    //bias?
    cbz     x3, activation_2   
    fadd v30.4s,v30.4s,v16.4s          //v1+bias
    fadd v31.4s,v31.4s,v16.4s         //v2+bias

    activation_2:
    cmp     w4,0
    blt     store_2

    fmax    v30.4s, v30.4s, v17.4s
    fmax    v31.4s, v31.4s, v17.4s
    beq     store_2

    store_2:
    // stp	 q30,q31, [x1, 0x20]
    st2  {v30.4s,v31.4s}, [x2]
return:
	ret
        .end

