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
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
*/


//x0: inp
//x1: out = inp_ptr + c*4 
//x2: ker
//x3: inw
//x4: inc_4*(sizeof(float))
//x5: inhw (to prefetch next channel)
    .section .text,"ax"
    .align 5

    .type tran_inp_4 STT_FUNC
    .global tran_inp_4
    .hidden tran_inp_4
    
tran_inp_4:
    sub	sp, sp, 0x40
    stp	d8, d9, [sp]
	stp	d10,d11,[sp, 0x10]
	stp	d12,d13,[sp, 0x20]
	stp	d14,d15,[sp, 0x30]
    
comput_idx:
    lsl	x3, x3, 0x2 
    add x11,x0,x3       
    add x12,x0,x3,LSL 1     
    add	x13,x11,x3, LSL 1  
    add	x14,x0,x3,LSL 2    

    lsl x15,x4,0x2
    lsl x16,x4,0x1
    add x17,x16,x15 //[1,0]=1*6+0

load:
    //v0 1 2 3 4 5 
    ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x11]
    ldr	q31, [x11, 0x40]	
    
    ext v4.16b,v0.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v5.16b,v1.16b,v31.16b,#4
    //v6 7 8 9 10 11
    lsl	x5, x5, 0x2 
    ld4 {v6.4s, v7.4s, v8.4s,v9.4s}, [x12]
    ldr	q31, [x12, 0x40]	
    ext v10.16b,v6.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v11.16b,v7.16b,v31.16b,#4
    add x6,x5,#0x40
    //v12 13 14 15 16 17 
    ld4 {v12.4s, v13.4s, v14.4s,v15.4s}, [x13]
    ldr	q31, [x13, 0x40]
    add x18,x1,x17	
    ext v16.16b,v12.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v17.16b,v13.16b,v31.16b,#4
    //v18 19 20 21 22 23
    ld4 {v18.4s, v19.4s, v20.4s,v21.4s}, [x14]
    ldr	q31, [x14, 0x40]	
    ext v22.16b,v18.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v23.16b,v19.16b,v31.16b,#4

    ldr	q30, [x2]

line1://add x18,x1,x17,[x18+=x4]
   
    fadd	v24.4s, v13.4s, v19.4s
    fadd	v25.4s, v14.4s, v20.4s
    fadd	v26.4s, v15.4s, v21.4s
    fadd	v27.4s, v16.4s, v22.4s
  
    fmls	v24.4s, v1.4s,  v30.s[2]
    fmls	v25.4s, v2.4s,  v30.s[2]
    prfm	pldl1keep, [x11, x5]
    fmls	v26.4s, v3.4s,  v30.s[2]
    fmls	v27.4s, v4.4s,  v30.s[2]

    movi	d28, 0x0
    fmls	v24.4s, v7.4s,  v30.s[2]
    fmls	v25.4s, v8.4s,  v30.s[2]
    fmls	v26.4s, v9.4s,  v30.s[2]
    fmls	v27.4s, v10.4s,  v30.s[2]
    
    //v29 as inp0
    fadd    v29.4s, v18.4s, v12.4s
    fmls	v28.4s, v25.4s,  v30.s[3]
    fadd	v31.4s, v26.4s,  v27.4s

    fmls	v29.4s, v0.4s,  v30.s[2]
    fmla	v28.4s, v27.4s,  v30.s[0]
    fmls	v29.4s, v6.4s,  v30.s[2]
    prfm	pldl1keep, [x11, x6]
    fmls	v31.4s, v24.4s,  v30.s[2]
    fmla	v28.4s, v29.4s,  v30.s[2]
  
    str q28, [x18]
    fmls	v31.4s, v25.4s,  v30.s[2]
    add x18,x18,x4
    fsub	v29.4s, v27.4s, v26.4s
    str q31, [x18]
	
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    fmls	v29.4s, v25.4s,  v30.s[2]
 
    str q29, [x18]
	movi	d28, 0x0
	movi	d29, 0x0
    fsub    v31.4s,v27.4s, v25.4s
    fmls	v28.4s, v24.4s,  v30.s[1]
    fadd    v25.4s, v17.4s,v23.4s

    fadd    v28.4s,v28.4s,v31.4s
    fmls	v25.4s, v5.4s,  v30.s[2]
    fmla	v31.4s, v24.4s,  v30.s[1]

    fmla	v28.4s, v26.4s,  v30.s[1]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    str q28, [x18] 
    fmls	v25.4s, v11.4s,  v30.s[2]
    fmls	v29.4s, v26.4s,  v30.s[3]
    fmls	v31.4s, v26.4s,  v30.s[1]
    add x18,x18,x4
    str q31, [x18]
    //v25 as inp0
    fmla	v29.4s, v25.4s,  v30.s[0]
    add x18,x18,x4
    str q29, [x18]

line2://add x18,x1,x17,LSL 1  (add x9,x18,x17)
    fsub    v24.4s, v19.4s, v13.4s
    fsub    v25.4s, v20.4s, v14.4s
    add x18,x1,x17,LSL 1 
    fsub    v26.4s, v21.4s, v15.4s
    fsub    v27.4s, v22.4s, v16.4s

    fmla	v24.4s, v1.4s,  v30.s[2]
    fmla	v25.4s, v2.4s,  v30.s[2] 
    add x9,x18,x17  
    fmla	v26.4s, v3.4s,  v30.s[2]
    fmla	v27.4s, v4.4s,  v30.s[2]
     movi	d28, 0x0
    fmls	v24.4s, v7.4s,  v30.s[2]
    fmls	v25.4s, v8.4s,  v30.s[2]
    prfm	pldl1keep, [x12, x5]
    fmls	v26.4s, v9.4s,  v30.s[2]
    fmls	v27.4s, v10.4s,  v30.s[2]

    //v29 as inp0
    fsub    v29.4s,v18.4s,v12.4s
    fmls	v28.4s, v25.4s,  v30.s[3]
    fadd	v31.4s, v26.4s,  v27.4s

    fmla	v29.4s, v0.4s,  v30.s[2]
    fmla	v28.4s, v27.4s,  v30.s[0]
    fmls	v29.4s, v6.4s,  v30.s[2]

    fmls	v31.4s, v24.4s,  v30.s[2]
    fmla	v28.4s, v29.4s,  v30.s[2]
    str q28, [x18]

    fmls	v31.4s, v25.4s,  v30.s[2]
    add x18,x18,x4
    fsub	v29.4s, v27.4s,  v26.4s

    str q31, [x18]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    fmls	v29.4s, v25.4s,  v30.s[2]
    str q29, [x18]
	movi	d28, 0x0
	movi	d29, 0x0
    fsub    v31.4s,v27.4s, v25.4s
    fmls	v28.4s, v24.4s,  v30.s[1]
    fsub    v25.4s,v23.4s,v17.4s
    fadd    v28.4s,v28.4s,v31.4s
    prfm	pldl1keep, [x12, x6]
    fmla	v25.4s, v5.4s,  v30.s[2]
    fmla	v31.4s, v24.4s,  v30.s[1]
    fmla	v28.4s, v26.4s,  v30.s[1]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    str q28, [x18]
    fmls	v25.4s, v11.4s,  v30.s[2]
    fmls	v29.4s, v26.4s,  v30.s[3]
    fmls	v31.4s, v26.4s,  v30.s[1]
    add x18,x18,x4
    str q31, [x18]
    //v25 as inp0
    fmla	v29.4s, v25.4s,  v30.s[0]
    add x18,x18,x4
    str q29, [x18]
line3://mov x18,x9 [x18+=x4]
    fsub	v24.4s, v19.4s,  v7.4s
    fsub	v25.4s, v20.4s,  v8.4s
    fsub	v26.4s, v21.4s,  v9.4s
    fsub	v27.4s, v22.4s,  v10.4s
    mov x18,x9
    fmls	v24.4s, v1.4s,  v30.s[1]
    fmls	v25.4s, v2.4s,  v30.s[1]
    fmls	v26.4s, v3.4s,  v30.s[1]
    fmls	v27.4s, v4.4s,  v30.s[1]
    movi	d28, 0x0
    fmla	v24.4s, v13.4s,  v30.s[1]
    fmla	v25.4s, v14.4s,  v30.s[1]
    prfm	pldl1keep, [x13, x5]
    fmla	v26.4s, v15.4s,  v30.s[1]
    fmla	v27.4s, v16.4s,  v30.s[1]


    //v29 as inp0
    fsub    v29.4s,v18.4s,v6.4s
    fmls	v28.4s, v25.4s,  v30.s[3]
    fadd	v31.4s, v26.4s,  v27.4s

    fmla	v29.4s, v12.4s,  v30.s[1]
    fmla	v28.4s, v27.4s,  v30.s[0]
    fmls	v29.4s, v0.4s,  v30.s[1]
    prfm	pldl1keep, [x13, x6]
    fmls	v31.4s, v24.4s,  v30.s[2]
    fmla	v28.4s, v29.4s,  v30.s[2]

    str q28, [x18]
    fmls	v31.4s, v25.4s,  v30.s[2]
    add x18,x18,x4
    fsub	v29.4s, v27.4s, v26.4s
    str q31, [x18]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    fmls	v29.4s, v25.4s,  v30.s[2]
    str q29, [x18]
	movi	d28, 0x0
	movi	d29, 0x0
    fsub    v31.4s,v27.4s, v25.4s
    fmls	v28.4s, v24.4s,  v30.s[1]
    fsub    v25.4s,v23.4s,v11.4s

    fadd    v28.4s,v28.4s,v31.4s
    fmla	v25.4s, v17.4s,  v30.s[1]
    fmla	v31.4s, v24.4s,  v30.s[1]

    fmla	v28.4s, v26.4s,  v30.s[1]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    str q28, [x18]

    fmls	v25.4s, v5.4s,  v30.s[1]
    fmls	v29.4s, v26.4s,  v30.s[3]
    fmls	v31.4s, v26.4s,  v30.s[1]

    add x18,x18,x4
    str q31, [x18]
    //v25 as inp0
    fmla	v29.4s, v25.4s,  v30.s[0]

    add x18,x18,x4
    str q29, [x18]

line4://add x18,x1,x17,LSL 2  ((add x9,x18,x17))
    fsub	v24.4s, v19.4s,  v7.4s
    fsub	v25.4s, v20.4s,  v8.4s
    fsub	v26.4s, v21.4s,  v9.4s
    fsub	v27.4s, v22.4s,  v10.4s
    add x18,x1,x17,LSL 2
    fmla	v24.4s, v1.4s,  v30.s[1]
    fmla	v25.4s, v2.4s,  v30.s[1]
    prfm	pldl1keep, [x14, x5]
    fmla	v26.4s, v3.4s,  v30.s[1]
    fmla	v27.4s, v4.4s,  v30.s[1]
    add x9,x18,x17
    fmls	v24.4s, v13.4s,  v30.s[1]
    fmls	v25.4s, v14.4s,  v30.s[1]
    movi	d28, 0x0    
    fmls	v26.4s, v15.4s,  v30.s[1]
    fmls	v27.4s, v16.4s,  v30.s[1]

    //v29 as inp0
    fsub    v29.4s,v18.4s,v6.4s
    fmla	v28.4s, v27.4s,  v30.s[0]
    fadd	v31.4s, v26.4s,  v27.4s

    fmla	v29.4s, V0.4s,  v30.s[1]
    fmls	v28.4s, v25.4s,  v30.s[3]
    fmls	v29.4s, v12.4s,  v30.s[1]

    fmls	v31.4s, v24.4s,  v30.s[2]
    fmla	v28.4s, v29.4s,  v30.s[2]
    prfm	pldl1keep, [x14, x6]
    str q28, [x18]

    //28 29 28 29
    fmls	v31.4s, v25.4s,  v30.s[2]
    add x18,x18,x4
    fsub	v29.4s, v27.4s,v26.4s

    str q31, [x18]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    fmls	v29.4s, v25.4s,  v30.s[2]
    str q29, [x18]
	movi	d28, 0x0
	movi	d29, 0x0
    fsub    v31.4s,v27.4s, v25.4s
    fmls	v28.4s, v24.4s,  v30.s[1]
    fsub    v25.4s,v23.4s,v11.4s
    fadd    v28.4s,v28.4s,v31.4s
    fmla	v25.4s, v5.4s,  v30.s[1]
    fmla	v31.4s, v24.4s,  v30.s[1]
    fmla	v28.4s, v26.4s,  v30.s[1]
    fmla	v29.4s, v24.4s,  v30.s[2]
    add x18,x18,x4
    str q28, [x18]
    fmls	v25.4s, v17.4s,  v30.s[1]
    fmls	v29.4s, v26.4s,  v30.s[3]
    fmls	v31.4s, v26.4s,  v30.s[1]
    add x18,x18,x4
    str q31, [x18]
    //v25 as inp0
    fmla	v29.4s, v25.4s,  v30.s[0]
    add x18,x18,x4
    str q29, [x18]

line0: // addr:   str q28, [x1], add x18,x1,x4
    //v0 1 2 3 4 5 
    ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x0]
    ldr	q31, [x0, 0x40]	
    ext v4.16b,v0.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v5.16b,v1.16b,v31.16b,#4
    movi	d29, 0x0
    movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
    fmla	v29.4s, V0.4s,  v30.s[2]
    fmla	v24.4s, v1.4s,  v30.s[2]
    movi	d28, 0x0
    movi	d31, 0x0
    fmla	v25.4s, v2.4s,  v30.s[2]
    fmls	v29.4s, v6.4s,  v30.s[3]
    prfm	pldl1keep, [x0, x5]
    fmla	v26.4s, v3.4s,  v30.s[2]
    fmla	v27.4s, v4.4s,  v30.s[2]    
    fmla	v29.4s, v18.4s,  v30.s[0]
    fmls	v24.4s, v7.4s,  v30.s[3]
    fmls	v25.4s, v8.4s,  v30.s[3]
    fmls	v26.4s, v9.4s,  v30.s[3]
    fmla	v28.4s, v29.4s,  v30.s[2]
   
    fmls	v27.4s, v10.4s,  v30.s[3]
    fmla	v25.4s, v20.4s,  v30.s[0]
    fmla	v24.4s, v19.4s,  v30.s[0]
    fmla	v27.4s, v22.4s,  v30.s[0]
    movi	d29, 0x0
    fmls	v28.4s, v25.4s,  v30.s[3]
    fmls	v31.4s, v24.4s,  v30.s[2]
    fmla	v26.4s, v21.4s,  v30.s[0]
    fmls	v31.4s, v25.4s,  v30.s[2]
    fmla	v29.4s, v24.4s,  v30.s[2]
    fmla	v28.4s, v27.4s,  v30.s[0]
    prfm	pldl1keep, [x0, x6]
    fmls	v29.4s, v25.4s,  v30.s[2]
    fmla	v31.4s, v26.4s,  v30.s[0]
    str q28, [x1]
    add x18,x1,x4
    fmla	v31.4s, v27.4s,  v30.s[0]
    movi	d28, 0x0
	fmls	v29.4s, v26.4s,  v30.s[0]
    fmls	v28.4s, v24.4s,  v30.s[1]
    str q31, [x18]
    fmla	v29.4s, v27.4s,  v30.s[0]
    fmla	v28.4s, v26.4s,  v30.s[1]
    add x18,x18,x4
    fsub    v31.4s, v27.4s,  v25.4s
    str q29, [x18]
    fadd	v28.4s, v28.4s,  v31.4s
    movi	d25, 0x0 
    movi	d29, 0x0
    add x18,x18,x4
    fmla	v25.4s, v5.4s,  v30.s[2]
    fmla	v29.4s, v24.4s,  v30.s[2]
    fmla	v31.4s, v24.4s,  v30.s[1]
    fmls	v25.4s, v11.4s,  v30.s[3]
    fmls	v29.4s, v26.4s,  v30.s[3]
    str q28, [x18]
    fmls	v31.4s, v26.4s,  v30.s[1]
    fmla	v25.4s, v23.4s,  v30.s[0]
    add x18,x18,x4
    fmla	v29.4s, v25.4s,  v30.s[0]
    str q31, [x18]
    add x18,x18,x4
    str q29, [x18]


line5://
    //v0 1 2 3 4 5 
    ld4 {v0.4s, v1.4s, v2.4s,v3.4s}, [x11]
    ldr	q31, [x11, 0x40]	
    ext v4.16b,v0.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v5.16b,v1.16b,v31.16b,#4

    //v18 19 20 21 22 23
    add x14,x14,x3     
    ld4 {v18.4s, v19.4s, v20.4s,v21.4s}, [x14]
    ldr	q31, [x14, 0x40]	
    ext v22.16b,v18.16b,v31.16b,#4
    ext v31.16b,v31.16b,v31.16b,#4
    ext v23.16b,v19.16b,v31.16b,#4

    movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0

    fmla	v24.4s, v1.4s,  v30.s[2]
    fmla	v25.4s, v2.4s,  v30.s[2]
    fmla	v26.4s, v3.4s,  v30.s[2]
    fmla	v27.4s, v4.4s,  v30.s[2]

    fmls	v24.4s, v13.4s,  v30.s[3]
    fmls	v25.4s, v14.4s,  v30.s[3]
    prfm	pldl1keep, [x14, x5]
    fmls	v26.4s, v15.4s,  v30.s[3]
    fmls	v27.4s, v16.4s,  v30.s[3]
    
    fmla	v24.4s, v19.4s,  v30.s[0]
    fmla	v25.4s, v20.4s,  v30.s[0]
    fmla	v26.4s, v21.4s,  v30.s[0]
    fmla	v27.4s, v22.4s,  v30.s[0]

    //v29 as inp0
     movi	d29, 0x0
    fmla	v29.4s, V0.4s,  v30.s[2]
    fmls	v29.4s, v12.4s,  v30.s[3]
    fmla	v29.4s, v18.4s,  v30.s[0]

    movi	d28, 0x0
    fmla	v28.4s, v29.4s,  v30.s[2]
    fmls	v28.4s, v25.4s,  v30.s[3]
    fmla	v28.4s, v27.4s,  v30.s[0]
    str q28, [x9]
 
    //28 29 28 29
    movi	d28, 0x0
	movi	d29, 0x0
    fmls	v28.4s, v24.4s,  v30.s[2]
    fmls	v28.4s, v25.4s,  v30.s[2]
    fmla	v28.4s, v26.4s,  v30.s[0]
    fmla	v28.4s, v27.4s,  v30.s[0]
    prfm	pldl1keep, [x14, x6]
    add x18,x9,x4
    str q28, [x18]
    fmla	v29.4s, v24.4s,  v30.s[2]
    fmls	v29.4s, v25.4s,  v30.s[2]
    fmls	v29.4s, v26.4s,  v30.s[0]
    fmla	v29.4s, v27.4s,  v30.s[0]
    add x18,x18,x4
    str q29, [x18]
	movi	d28, 0x0
	movi	d29, 0x0
    fmls	v28.4s, v24.4s,  v30.s[1]
    fmls	v28.4s, v25.4s,  v30.s[0]
    fmla	v28.4s, v26.4s,  v30.s[1]
    fmla	v28.4s, v27.4s,  v30.s[0]
    add x18,x18,x4
    str q28, [x18]
    fmla	v29.4s, v24.4s,  v30.s[1]
    fmls	v29.4s, v25.4s,  v30.s[0]
    fmls	v29.4s, v26.4s,  v30.s[1]
    fmla	v29.4s, v27.4s,  v30.s[0]
    add x18,x18,x4
    str q29, [x18]

        //v25 as inp0
     movi	d29, 0x0   
 
    fmla	v29.4s, v5.4s,  v30.s[2]
    fmls	v29.4s, v17.4s,  v30.s[3]
    fmla	v29.4s, v23.4s,  v30.s[0]
    movi	d28, 0x0
    fmla	v28.4s, v24.4s,  v30.s[2]
    fmls	v28.4s, v26.4s,  v30.s[3]
    fmla	v28.4s, v29.4s,  v30.s[0]
    add x18,x18,x4
    str q28, [x18]
//
return:
	ldp	d8,  d9,  [sp]
	ldp	d10, d11, [sp, 0x10]
	ldp	d12, d13, [sp, 0x20]
	ldp	d14, d15, [sp, 0x30]
	add	sp, sp, 0x40
	ret
        .end


//stp  q24,q25, [x1]
//stp	 q26,q27, [x1, 0x20]
//stp	 q28,q29, [x1, 0x40]
