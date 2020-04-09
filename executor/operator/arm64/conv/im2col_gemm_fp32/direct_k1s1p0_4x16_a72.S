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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
 
// register definition
//------------------------------
// x0        bias start address
// x1        input start address
// x2        kernel start address
// x3        output start address
// x4        in_hw
// x5        c_in
// x6         activation
//--------------------------------
// x9        cin/4

//--------------------------------
// x1        inp               
// x12       inp+ hw
// x13       inp+ 2*hw
// x14       inp+ 3*hwx6
// preload for 3 block[4 lines of cin]
//=============================
// input q0 q1 
// kernel q4 q5 q6 q7
// output q16-q30


    .section .text,"ax"
    .align 5

    .type direct_k1s1p0_4x16_a72 STT_FUNC
    .global direct_k1s1p0_4x16_a72
    .hidden direct_k1s1p0_4x16_a72

	
direct_k1s1p0_4x16_a72:
    prfm	pldl1keep, [x1]
    lsl x4,x4,2                    
    prfm	pldl1keep, [x2]
    cbz    x0, none_biases            
        ld4r    {v16.4s,v17.4s,v18.4s,v19.4s}, [x0], 0x10
        ld4r    {v20.4s,v21.4s,v22.4s,v23.4s}, [x0], 0x10
        ld4r    {v24.4s,v25.4s,v26.4s,v27.4s}, [x0], 0x10
        ld4r    {v28.4s,v29.4s,v30.4s,v31.4s}, [x0]
        b convolution_start

none_biases:
    movi    d16, 0x0
    movi    d17, 0x0
    movi    d18, 0x0
    movi    d19, 0x0
    movi    d20, 0x0
    movi    d21, 0x0
    movi    d22, 0x0
    movi    d23, 0x0
    movi    d24, 0x0
    movi    d25, 0x0
    movi    d26, 0x0
    movi    d27, 0x0
    movi    d28, 0x0
    movi    d29, 0x0
    movi    d30, 0x0
    movi    d31, 0x0

convolution_start:

    cmp    x5,0x4                        
    b.lt    loop4_end
   
    add    x12, x1, x4                     // inp2 = inp + hw *sizeof(float)           
    lsr    x9, x5, 0x2                   
    add    x13, x12, x4                    // inp3 = inp + hw * 2 *sizeof(float)
    add    x14, x13, x4                    // inp4 = inp + hw * 3 *sizeof(float)
    mov    x7,#12
    mul    x7,x7,x4
    
loop4:  
    ldr    q0, [x1]                                                   
    ldr    q4,  [x2]    
	fmla	v16.4s, v0.4s,  v4.s[0]	
	fmla	v17.4s, v0.4s,  v4.s[1]	
    ldr    q5,  [x2,0x10]  
	fmla	v18.4s, v0.4s,  v4.s[2]	
	fmla	v19.4s, v0.4s,  v4.s[3]	
	fmla	v20.4s, v0.4s,  v5.s[0]	
	fmla	v21.4s, v0.4s,  v5.s[1]	
    ldp	q6, q7, [x2, 0x20]		 
	subs	x9, x9, 0x1
	fmla	v22.4s, v0.4s,  v5.s[2]	
	fmla	v23.4s, v0.4s,  v5.s[3]	
    ldr	q1, [x12]		        	
	fmla	v24.4s, v0.4s,  v6.s[0]	
	fmla	v25.4s, v0.4s,  v6.s[1]	
	fmla	v26.4s, v0.4s,  v6.s[2]	
	fmla	v27.4s, v0.4s,  v6.s[3]	
    ldp	q4, q5, [x2, 0x40]		  
	prfm	pldl1keep, [x1, x7]
	fmla	v28.4s, v0.4s,  v7.s[0]	
	fmla	v29.4s, v0.4s,  v7.s[1]	
	prfm	pldl1keep, [x2, 0x140]
	fmla	v30.4s, v0.4s,  v7.s[2]	
	fmla	v31.4s, v0.4s,  v7.s[3]	

	ldp	q6, q7, [x2, 0x60]		
	fmla	v16.4s, v1.4s,  v4.s[0]	
	fmla	v17.4s, v1.4s,  v4.s[1]	
	fmla	v18.4s, v1.4s,  v4.s[2]	
	fmla	v19.4s, v1.4s,  v4.s[3]	
	ldr	q0, [x13]			
	fmla	v20.4s, v1.4s,  v5.s[0]	
	fmla	v21.4s, v1.4s,  v5.s[1]	
	fmla	v22.4s, v1.4s,  v5.s[2]	
	fmla	v23.4s, v1.4s,  v5.s[3]	
	ldp	q4, q5, [x2, 0x80]		
	fmla	v24.4s, v1.4s,  v6.s[0]	
	fmla	v25.4s, v1.4s,  v6.s[1]	
	fmla	v26.4s, v1.4s,  v6.s[2]	
	fmla	v27.4s, v1.4s,  v6.s[3]	
	prfm	pldl1keep, [x2, 0x180]
	fmla	v28.4s, v1.4s,  v7.s[0]	
	fmla	v29.4s, v1.4s,  v7.s[1]	
	prfm	pldl1keep, [x2, 0x1c0]
	fmla	v30.4s, v1.4s,  v7.s[2]	
	fmla	v31.4s, v1.4s,  v7.s[3]	

	ldp	q6, q7, [x2,0xa0]		
	fmla	v16.4s, v0.4s,  v4.s[0]	
	fmla	v17.4s, v0.4s,  v4.s[1]	
	fmla	v18.4s, v0.4s,  v4.s[2]	
	fmla	v19.4s, v0.4s,  v4.s[3]	

	fmla	v20.4s, v0.4s,  v5.s[0]	
	fmla	v21.4s, v0.4s,  v5.s[1]	
	fmla	v22.4s, v0.4s,  v5.s[2]	
	fmla	v23.4s, v0.4s,  v5.s[3]	
	ldp	q4, q5, [x2, 0xc0]		
	fmla	v24.4s, v0.4s,  v6.s[0]	
	fmla	v25.4s, v0.4s,  v6.s[1]	
    add    x1,x1, x4,LSL 2 
    ldr	q1, [x14]			
	fmla	v26.4s, v0.4s,  v6.s[2]	
	fmla	v27.4s, v0.4s,  v6.s[3]	
	prfm	pldl1keep, [x2, 0x200]
	fmla	v28.4s, v0.4s,  v7.s[0]	
	fmla	v29.4s, v0.4s,  v7.s[1]	
     prfm    pldl1keep, [x12, x7]
	fmla	v30.4s, v0.4s,  v7.s[2]	
	fmla	v31.4s, v0.4s,  v7.s[3]	

	ldp	q6, q7, [x2, 0xe0]		
	fmla	v16.4s, v1.4s,  v4.s[0]	
	fmla	v17.4s, v1.4s,  v4.s[1]	
	add	x2, x2, 0x100
	fmla	v18.4s, v1.4s,  v4.s[2]	
	fmla	v19.4s, v1.4s,  v4.s[3]	
	
    add    x12,x12, x4,LSL 2 
	fmla	v20.4s, v1.4s,  v5.s[0]	
	fmla	v21.4s, v1.4s,  v5.s[1]	
    prfm    pldl1keep, [x13, x7]
	fmla	v22.4s, v1.4s,  v5.s[2]	
	fmla	v23.4s, v1.4s,  v5.s[3]	
	add    x13,x13, x4,LSL 2 
	fmla	v24.4s, v1.4s,  v6.s[0]	
	fmla	v25.4s, v1.4s,  v6.s[1]	
    prfm    pldl1keep, [x14, x7]
	fmla	v26.4s, v1.4s,  v6.s[2]	
	fmla	v27.4s, v1.4s,  v6.s[3]	
    add    x14,x14, x4,LSL 2 
	fmla	v28.4s, v1.4s,  v7.s[0]	
	fmla	v29.4s, v1.4s,  v7.s[1]	
	fmla	v30.4s, v1.4s,  v7.s[2]	
	fmla	v31.4s, v1.4s,  v7.s[3]	
	b.ne	loop4

loop4_end:
    and    x11,x5,0x3 
   
    lsl	x14,x4, 2
    cbz    x11, activation

loop1:
    ldr    q0, [x1]   // inp0   
    
    ldp     q4, q5, [x2]                  
    
    fmla    v16.4s, v0.4s,  v4.s[0]    
    fmla    v17.4s, v0.4s,  v4.s[1]    
    fmla    v18.4s, v0.4s,  v4.s[2]    
    fmla    v19.4s, v0.4s,  v4.s[3]    
    ldp     q6, q7, [x2, 0x20]
    fmla    v20.4s, v0.4s,  v5.s[0]    
    fmla    v21.4s, v0.4s,  v5.s[1]    
    fmla    v22.4s, v0.4s,  v5.s[2]    
    fmla    v23.4s, v0.4s,  v5.s[3]    
    subs    x11,x11,0x1
    prfm    pldl1keep, [x2, 0x80]
    fmla    v24.4s, v0.4s,  v6.s[0]    
    fmla    v25.4s, v0.4s,  v6.s[1]    
    fmla    v26.4s, v0.4s,  v6.s[2]    
    fmla    v27.4s, v0.4s,  v6.s[3]    
       add    x1, x1, x4    
    fmla    v28.4s, v0.4s,  v7.s[0]    
    fmla    v29.4s, v0.4s,  v7.s[1]    
    fmla    v30.4s, v0.4s,  v7.s[2]    
    fmla    v31.4s, v0.4s,  v7.s[3]
    add    x2, x2, 0x40
    
    b.ne loop1

activation:
    cmp w6,0
    blt save_result

    movi	d0, 0
	scvtf   s1,w6

	fmax	v16.4s, v16.4s, v0.4s
	fmax	v17.4s, v17.4s, v0.4s
	fmax	v18.4s, v18.4s, v0.4s
	fmax	v19.4s, v19.4s, v0.4s
	fmax	v20.4s, v20.4s, v0.4s
	fmax	v21.4s, v21.4s, v0.4s
	fmax	v22.4s, v22.4s, v0.4s
	fmax	v23.4s, v23.4s, v0.4s
	fmax	v24.4s, v24.4s, v0.4s
	fmax	v25.4s, v25.4s, v0.4s
	fmax	v26.4s, v26.4s, v0.4s
	fmax	v27.4s, v27.4s, v0.4s
	fmax	v28.4s, v28.4s, v0.4s
	fmax	v29.4s, v29.4s, v0.4s
	fmax	v30.4s, v30.4s, v0.4s
	fmax	v31.4s, v31.4s, v0.4s

	beq  save_result

	dup     v1.4s,v1.s[0]

	fmin	v16.4s, v16.4s, v1.4s
	fmin	v17.4s, v17.4s, v1.4s
	fmin	v18.4s, v18.4s, v1.4s
	fmin	v19.4s, v19.4s, v1.4s
	fmin	v20.4s, v20.4s, v1.4s
	fmin	v21.4s, v21.4s, v1.4s
	fmin	v22.4s, v22.4s, v1.4s
	fmin	v23.4s, v23.4s, v1.4s
	fmin	v24.4s, v24.4s, v1.4s
	fmin	v25.4s, v25.4s, v1.4s
	fmin	v26.4s, v26.4s, v1.4s
	fmin	v27.4s, v27.4s, v1.4s
	fmin	v28.4s, v28.4s, v1.4s
	fmin	v29.4s, v29.4s, v1.4s
	fmin	v30.4s, v30.4s, v1.4s
	fmin	v31.4s, v31.4s, v1.4s


save_result:
    add x11,x3,x4
	add	x12,x3, x4, LSL 1		
	add	x13,x11,x4, LSL 1		
    
    str     q16, [x3]
	add	x3, x3, x14
	str	q17, [x11]
	add	x11,x11, x14
	str	q18, [x12]
	add	x12,x12, x14
	str	q19, [x13]
	add	x13,x13, x14

    str     q20, [x3]
	add	x3, x3, x14
	str	q21, [x11]
	add	x11,x11,x14
	str	q22, [x12]
	add	x12,x12,x14
	str	q23, [x13]
	add	x13,x13,x14

    str     q24, [x3]
	add	x3, x3, x14
	str	q25, [x11]
	add	x11,x11,x14
	str	q26, [x12]
	add	x12,x12,x14
	str	q27, [x13]
	add	x13,x13,x14

    str     q28, [x3]
	str	q29, [x11]
	str	q30, [x12]
	str	q31, [x13]

    ret
       .end

