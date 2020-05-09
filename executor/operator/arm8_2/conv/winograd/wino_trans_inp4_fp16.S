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
 * Unless reduired by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/


//x0: inp
//x1: out = inp_ptr + c*4 
//x2: ker
//x3: inw
//x4: inc_4*(sizeof(float16))
//x5: inhw (to prefetch next channel)
    .section .text,"ax"
    .align 5

    .type wino_trans_inp4_fp16 STT_FUNC
    .global wino_trans_inp4_fp16
    .hidden wino_trans_inp4_fp16
    
wino_trans_inp4_fp16:
    sub	sp, sp, 0x40
    stp	d8, d9, [sp]
	stp	d10,d11,[sp, 0x10]
	stp	d12,d13,[sp, 0x20]
	stp	d14,d15,[sp, 0x30]
    
comput_idx:
    lsl	x3, x3, 0x1                   // inw * sizeof(fp16)
    add x11,x0,x3                     // inp + inw
    add x12,x0,x3,LSL 1               // inp + 2* inw
    add	x13,x11,x3, LSL 1             // inp + 3* inw
    add	x14,x0,x3,LSL 2               // inp + 4* inw

    lsl x15,x4,0x2                    //x15=inc_4 * 4
    lsl x16,x4,0x1                    //x16 =inc_4 *2
    add x17,x16,x15 //[1,0]=1*6+0     //x17=inc_4*6

load:
    //load v0-v24

    //v0 1 2 3 4 5 
    ld4 {v0.4h, v1.4h, v2.4h,v3.4h}, [x11]
    ldr	d31, [x11, 0x20]	
    ext v4.8b,v0.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v30.8b,v1.8b,v31.8b,#2

    //v6 7 8 9 10 11
    lsl	x5, x5, 0x1                   //inp_hw * sizeof(fp16)
    ld4 {v6.4h, v7.4h, v8.4h,v9.4h}, [x12]
    ldr	d31, [x12, 0x20]	
    ext v10.8b,v6.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v11.8b,v7.8b,v31.8b,#2

    //v12 13 14 15 16 17 
    ld4 {v12.4h, v13.4h, v14.4h,v15.4h}, [x13]
    ldr	d31, [x13, 0x20]
    add x18,x1,x17	
    ext v16.8b,v12.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v17.8b,v13.8b,v31.8b,#2

    //v18 19 20 21 22 23
    ld4 {v18.4h, v19.4h, v20.4h,v21.4h}, [x14]
    ldr	d31, [x14, 0x20]	
    ext v22.8b,v18.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v23.8b,v19.8b,v31.8b,#2

    ldr	q5, [x2] 

    fmul v0.4h,v0.4h,v5.h[0]  
    fmul v1.4h,v1.4h,v5.h[0]  
    fmul v2.4h,v2.4h,v5.h[0]
    fmul v3.4h,v3.4h,v5.h[0]
    fmul v4.4h,v4.4h,v5.h[0]
    fmul v30.4h,v30.4h,v5.h[0]

    fmul v6.4h,v6.4h,v5.h[0]  
    fmul v7.4h,v7.4h,v5.h[0]  
    fmul v8.4h,v8.4h,v5.h[0]
    fmul v9.4h,v9.4h,v5.h[0]
    fmul v10.4h,v10.4h,v5.h[0]
    fmul v11.4h,v11.4h,v5.h[0]

    fmul v12.4h,v12.4h,v5.h[0]
    fmul v13.4h,v13.4h,v5.h[0]
    fmul v14.4h,v14.4h,v5.h[0]
    fmul v15.4h,v15.4h,v5.h[0]  
    fmul v16.4h,v16.4h,v5.h[0]
    fmul v17.4h,v17.4h,v5.h[0]

    fmul v18.4h,v18.4h,v5.h[0]
    fmul v19.4h,v19.4h,v5.h[0]
    fmul v20.4h,v20.4h,v5.h[0]
    fmul v21.4h,v21.4h,v5.h[0]
    fmul v22.4h,v22.4h,v5.h[0]
    fmul v23.4h,v23.4h,v5.h[0]

line1://add x18,x1,x17,[x18+=x4]

    fadd    v29.4h, v12.4h, v18.4h
    fadd	v24.4h, v13.4h, v19.4h
    fadd	v25.4h, v14.4h, v20.4h
    fadd	v26.4h, v15.4h, v21.4h
    fadd	v27.4h, v16.4h, v22.4h
  
    fmls	v29.4h, v0.4h,  v5.h[6]
    fmls	v24.4h, v1.4h,  v5.h[6]
    fmls	v25.4h, v2.4h,  v5.h[6]
    fmls	v26.4h, v3.4h,  v5.h[6]
    fmls	v27.4h, v4.4h,  v5.h[6]
    prfm	pldl1keep, [x11, x5]

    movi	d31, 0x0
    fmls	v29.4h, v6.4h,  v5.h[6]
    fmls	v24.4h, v7.4h,  v5.h[6] 
    fmls	v25.4h, v8.4h,  v5.h[6]
    fmls	v26.4h, v9.4h,  v5.h[6]
    fmls	v27.4h, v10.4h, v5.h[6]


    fmla    v31.4h,v29.4h,v5.h[2]
    fadd   v28.4h, v26.4h,v27.4h
    fsub   v29.4h, v27.4h,v26.4h

    fmls    v31.4h,v25.4h,v5.h[3]
    fmla    v31.4h,v27.4h,v5.h[0]
    str d31, [x18]

    fmls   v28.4h, v24.4h,v5.h[6]
    fmls   v28.4h, v25.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    fmla   v29.4h, v24.4h,v5.h[6]
    fmls   v29.4h, v25.4h,v5.h[6]
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]

    fsub   v31.4h, v27.4h,v25.4h
    fmla   v31.4h, v26.4h,v5.h[5]
    fmls   v31.4h, v24.4h,v5.h[5]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

    fsub   v28.4h, v27.4h,v25.4h
    fmls   v28.4h, v26.4h,v5.h[5]
    fmla   v28.4h, v24.4h,v5.h[5]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    //v25 as inp0
    fadd    v25.4h, v23.4h, v17.4h 
    fmls	v25.4h, v30.4h,  v5.h[6]
    fmls	v25.4h, v11.4h,  v5.h[6]

    movi	d31, 0x0
    fmla   v31.4h, v24.4h,v5.h[2]
    fmls   v31.4h, v26.4h,v5.h[3]
    fmla   v31.4h, v25.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

line2://add x18,x1,x17,LSL 1  (add x9,x18,x17)
    fsub    v29.4h, v18.4h, v12.4h 
    fsub    v24.4h, v19.4h, v13.4h
    fsub    v25.4h, v20.4h, v14.4h
    add x18,x1,x17,LSL 1 
    fsub    v26.4h, v21.4h, v15.4h
    fsub    v27.4h, v22.4h, v16.4h

    fmla	v29.4h, v0.4h,  v5.h[6]
    fmla	v24.4h, v1.4h,  v5.h[6]
    fmla	v25.4h, v2.4h,  v5.h[6] 
    add x9,x18,x17  
    fmla	v26.4h, v3.4h,  v5.h[6]
    fmla	v27.4h, v4.4h,  v5.h[6]

    movi	d31, 0x0
    fmls	v29.4h, v6.4h,  v5.h[6]
    fmls	v24.4h, v7.4h,  v5.h[6]
    fmls	v25.4h, v8.4h,  v5.h[6]
    prfm	pldl1keep, [x12, x5]
    fmls	v26.4h, v9.4h,  v5.h[6]
    fmls	v27.4h, v10.4h,  v5.h[6]

    fmla    v31.4h,v29.4h,v5.h[2]
    fadd   v28.4h, v26.4h,v27.4h
    fsub   v29.4h, v27.4h,v26.4h

    fmls    v31.4h,v25.4h,v5.h[3]
    fmla    v31.4h,v27.4h,v5.h[0]
    str d31, [x18]

    fmls   v28.4h, v24.4h,v5.h[6]
    fmls   v28.4h, v25.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    fmla   v29.4h, v24.4h,v5.h[6]
    fmls   v29.4h, v25.4h,v5.h[6]
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]

    fsub   v31.4h, v27.4h,v25.4h
    fmla   v31.4h, v26.4h,v5.h[5]
    fmls   v31.4h, v24.4h,v5.h[5]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

    fsub   v28.4h, v27.4h,v25.4h
    fmls   v28.4h, v26.4h,v5.h[5]
    fmla   v28.4h, v24.4h,v5.h[5]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    //v25 as inp0
    fsub    v25.4h, v23.4h, v17.4h 
    fmla	v25.4h, v30.4h,  v5.h[6]
    fmls	v25.4h, v11.4h,  v5.h[6]

    movi	d31, 0x0
    fmla   v31.4h, v24.4h,v5.h[2]
    fmls   v31.4h, v26.4h,v5.h[3]
    fmla   v31.4h, v25.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

line3://mov x18,x9 [x18+=x4]
    fsub    v29.4h, v18.4h, v6.4h 
    fsub	v24.4h, v19.4h,  v7.4h
    fsub	v25.4h, v20.4h,  v8.4h
    fsub	v26.4h, v21.4h,  v9.4h
    fsub	v27.4h, v22.4h,  v10.4h

    mov x18,x9
    fmls	v29.4h, v0.4h,  v5.h[5]
    fmls	v24.4h, v1.4h,  v5.h[5]
    fmls	v25.4h, v2.4h,  v5.h[5]
    fmls	v26.4h, v3.4h,  v5.h[5]
    fmls	v27.4h, v4.4h,  v5.h[5]

    movi	d31, 0x0
    fmla	v29.4h, v12.4h,  v5.h[5]
    fmla	v24.4h, v13.4h,  v5.h[5]
    fmla	v25.4h, v14.4h,  v5.h[5]
    prfm	pldl1keep, [x13, x5]
    fmla	v26.4h, v15.4h,  v5.h[5]
    fmla	v27.4h, v16.4h,  v5.h[5]

    fmla    v31.4h,v29.4h,v5.h[2]
    fadd   v28.4h, v26.4h,v27.4h
    fsub   v29.4h, v27.4h,v26.4h

    fmls    v31.4h,v25.4h,v5.h[3]
    fmla    v31.4h,v27.4h,v5.h[0]
    str d31, [x18]

    fmls   v28.4h, v24.4h,v5.h[6]
    fmls   v28.4h, v25.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    fmla   v29.4h, v24.4h,v5.h[6]
    fmls   v29.4h, v25.4h,v5.h[6]
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]

    fsub   v31.4h, v27.4h,v25.4h
    fmla   v31.4h, v26.4h,v5.h[5]
    fmls   v31.4h, v24.4h,v5.h[5]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

    fsub   v28.4h, v27.4h,v25.4h
    fmls   v28.4h, v26.4h,v5.h[5]
    fmla   v28.4h, v24.4h,v5.h[5]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    //v25 as inp0
    fsub    v25.4h, v23.4h, v11.4h 
    fmls	v25.4h, v30.4h,  v5.h[5]
    fmla	v25.4h, v17.4h,  v5.h[5]

    movi	d31, 0x0
    fmla   v31.4h, v24.4h,v5.h[2]
    fmls   v31.4h, v26.4h,v5.h[3]
    fmla   v31.4h, v25.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]



line4://add x18,x1,x17,LSL 2  ((add x9,x18,x17))
    fsub    v29.4h, v18.4h, v6.4h
    fsub	v24.4h, v19.4h,  v7.4h
    fsub	v25.4h, v20.4h,  v8.4h
    fsub	v26.4h, v21.4h,  v9.4h
    fsub	v27.4h, v22.4h,  v10.4h
    add x18,x1,x17,LSL 2

    fmla	v29.4h, v0.4h,  v5.h[5]
    fmla	v24.4h, v1.4h,  v5.h[5]
    fmla	v25.4h, v2.4h,  v5.h[5]
    prfm	pldl1keep, [x14, x5]
    fmla	v26.4h, v3.4h,  v5.h[5]
    fmla	v27.4h, v4.4h,  v5.h[5]
    add x9,x18,x17

    movi	d31, 0x0
    fmls	v29.4h, v12.4h,  v5.h[5]
    fmls	v24.4h, v13.4h,  v5.h[5]
    fmls	v25.4h, v14.4h,  v5.h[5]
    fmls	v26.4h, v15.4h,  v5.h[5]
    fmls	v27.4h, v16.4h,  v5.h[5]

    fmla    v31.4h,v29.4h,v5.h[2]
    fadd   v28.4h, v26.4h,v27.4h
    fsub   v29.4h, v27.4h,v26.4h

    fmls    v31.4h,v25.4h,v5.h[3]
    fmla    v31.4h,v27.4h,v5.h[0]
    str d31, [x18]

    fmls   v28.4h, v24.4h,v5.h[6]
    fmls   v28.4h, v25.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    fmla   v29.4h, v24.4h,v5.h[6]
    fmls   v29.4h, v25.4h,v5.h[6]
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]

    fsub   v31.4h, v27.4h,v25.4h
    fmla   v31.4h, v26.4h,v5.h[5]
    fmls   v31.4h, v24.4h,v5.h[5]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

    fsub   v28.4h, v27.4h,v25.4h
    fmls   v28.4h, v26.4h,v5.h[5]
    fmla   v28.4h, v24.4h,v5.h[5]
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]

    //v25 as inp0
    fsub    v25.4h, v23.4h, v11.4h 
    fmla	v25.4h, v30.4h,  v5.h[5]
    fmls	v25.4h, v17.4h,  v5.h[5]

    movi	d31, 0x0
    fmla   v31.4h, v24.4h,v5.h[2]
    fmls   v31.4h, v26.4h,v5.h[3]
    fmla   v31.4h, v25.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]

line0: // addr:   str d28, [x1], add x18,x1,x4
    //line0
    ld4 {v24.4h, v25.4h, v26.4h,v27.4h}, [x0]
    ldr	d31, [x0, 0x20]	
    ext v28.8b,v24.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v29.8b,v25.8b,v31.8b,#2

    fmla   v18.4h, v24.4h,v5.h[2]
    fmla   v19.4h, v25.4h,v5.h[2]
    fmla   v20.4h, v26.4h,v5.h[2]
    fmla   v21.4h, v27.4h,v5.h[2]
    fmla   v22.4h, v28.4h,v5.h[2]
    fmla   v23.4h, v29.4h,v5.h[2]

    prfm	pldl1keep, [x0, x5]
    fmls   v18.4h, v6.4h,v5.h[7]
    fmls   v19.4h, v7.4h,v5.h[7]
    fmls   v20.4h, v8.4h,v5.h[7]
    fmls   v21.4h, v9.4h,v5.h[7]
    fmls   v22.4h, v10.4h,v5.h[7]
    fmls   v23.4h, v11.4h,v5.h[7]
    //end mid

    fmul    v18.4h,v18.4h,v5.h[2]
    fmls    v18.4h,v20.4h,v5.h[3]
    fmla    v18.4h,v22.4h,v5.h[0]
    str d18, [x1]
    add x18,x1,x4                  //[0]

    fadd   v28.4h, v21.4h,v22.4h
    fmls   v28.4h, v19.4h,v5.h[6]
    fmls   v28.4h, v20.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    str d28, [x18]               //[1]

    fsub   v31.4h, v22.4h,v21.4h
    fmla   v31.4h, v19.4h,v5.h[6]
    fmls   v31.4h, v20.4h,v5.h[6]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]               //[2]

    fsub   v25.4h, v22.4h,v20.4h
    fsub   v26.4h, v21.4h,v19.4h
    fmul   v26.4h,v26.4h,v5.h[5]

    fadd   v29.4h,v25.4h,v26.4h
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]                  //[3]

    fsub   v28.4h, v25.4h,v26.4h
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]                  //[4]

    fmul   v23.4h,v23.4h, v5.h[0]
    fmla   v23.4h, v19.4h,v5.h[2]
    fmls   v23.4h, v21.4h,v5.h[3]
    add x18,x18,x4
    str d23, [x18]


line5: 
    //line5
    add x14,x14,x3     
    ld4 {v18.4h, v19.4h, v20.4h,v21.4h}, [x14]
    ldr	d31, [x14, 0x20]	
    ext v22.8b,v18.8b,v31.8b,#2
    ext v31.8b,v31.8b,v31.8b,#2
    ext v23.8b,v19.8b,v31.8b,#2

    fmul  v18.4h,v18.4h,v5.h[0]  
    fmul  v19.4h,v19.4h,v5.h[0]
    fmul  v20.4h,v20.4h,v5.h[0]
    fmul  v21.4h,v21.4h,v5.h[0]
    fmul  v22.4h,v22.4h,v5.h[0]
    fmul  v23.4h,v23.4h,v5.h[0]

    fmla   v18.4h, v0.4h,v5.h[6]
    fmla   v19.4h, v1.4h,v5.h[6]
    fmla   v20.4h, v2.4h,v5.h[6]
    fmla   v21.4h, v3.4h,v5.h[6]
    fmla   v22.4h, v4.4h,v5.h[6]
    fmla   v23.4h, v30.4h,v5.h[6]
    prfm	pldl1keep, [x14, x5]

    fmls   v18.4h, v12.4h,v5.h[7]
    fmls   v19.4h, v13.4h,v5.h[7]
    fmls   v20.4h, v14.4h,v5.h[7]
    fmls   v21.4h, v15.4h,v5.h[7]
    fmls   v22.4h, v16.4h,v5.h[7]
    fmls   v23.4h, v17.4h,v5.h[7]
    //end mid

    fmul    v18.4h,v18.4h,v5.h[2]
    fmls    v18.4h,v20.4h,v5.h[3]
    fmla    v18.4h,v22.4h,v5.h[0]
    str d18, [x9]
    add x18,x9,x4                  //[0]

    fadd   v28.4h, v21.4h,v22.4h
    fmls   v28.4h, v19.4h,v5.h[6]
    fmls   v28.4h, v20.4h,v5.h[6]
    fmul   v28.4h, v28.4h,v5.h[0]
    str d28, [x18]               //[1]

    fsub   v31.4h, v22.4h,v21.4h
    fmla   v31.4h, v19.4h,v5.h[6]
    fmls   v31.4h, v20.4h,v5.h[6]
    fmul   v31.4h, v31.4h,v5.h[0]
    add x18,x18,x4
    str d31, [x18]               //[2]

    fsub   v25.4h, v22.4h,v20.4h
    fsub   v26.4h, v21.4h,v19.4h
    fmul   v26.4h,v26.4h,v5.h[5]

    fadd   v29.4h,v25.4h,v26.4h
    fmul   v29.4h, v29.4h,v5.h[0]
    add x18,x18,x4
    str d29, [x18]                  //[3]

    fsub   v28.4h, v25.4h,v26.4h
    fmul   v28.4h, v28.4h,v5.h[0]
    add x18,x18,x4
    str d28, [x18]                  //[4]

    fmul   v23.4h,v23.4h, v5.h[0]
    fmla   v23.4h, v19.4h,v5.h[2]
    fmls   v23.4h, v21.4h,v5.h[3]
    add x18,x18,x4
    str d23, [x18]


return:
	ldp	d8,  d9,  [sp]
	ldp	d10, d11, [sp, 0x10]
	ldp	d12, d13, [sp, 0x20]
	ldp	d14, d15, [sp, 0x30]
	add	sp, sp, 0x40
	ret
        .end


//stp  d24,d25, [x1]
//stp	 d26,d27, [x1, 0x20]
//stp	 d28,d29, [x1, 0x40]
