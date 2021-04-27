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
//x0: input
//x1: h
//x2: w
//x3: kernel
//x4: output //L-2
//x5: bias
//x6: output_w
//x7: output_w * sizeof(float)
//x8: output L1

//x9: width * sizeof(float)
//x10: input point tmp L0
//x11: input_h tmp
//x12:  input point tmp L1

//x13: input_w tmp
//x14: input_w tmp
//x15 && x16: for pld
//x17: output point L0

//input: v0 ~ v4
//shift: v5,v13,v27,v28
//input: v6~v10
//kernel vector: v24 ~ v26
//output_vec  v14~v17
//bias: v21
//zero vector : v31
//activation:  x18
//activation vector: v29

#define CONV_RELU_FUSE

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p0p1
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:

   mov x18,x7
   scvtf s29,w18
   dup v29.4s,v29.s[0]

   //load kernel
   ldp q24,q25,[x3]
   ldr s26,[x3,#32]
   
   ext  v26.16b,v25.16b,v26.16b,8
   ext  v25.16b,v24.16b,v25.16b,12
   
   lsl x9,x2,2       //input_w * sizeof(float)
   lsl x7,x6,2       // output_w * sizeof(float)  
   
   mov  x15,#8
   mul  x15,x2,x15 
   mov x13,#0
   mov x11,#0
   movi d31,#0
   sub sp,sp,#0x40
   stp d8,d9,[sp]
   stp d10,d11,[sp,0x10]
   stp d12,d13,[sp,0x20]
   stp d14,d15,[sp,0x30]

   //get bias
   cbz x5,non_biases
   ld1r {v21.4s}, [x5]  
   b first_row_3_start

non_biases:
   movi d21,#0


first_row_3_start: 
    movi d14,#0
    movi d15,#0
    mov x10,x0
    mov x8,x4
    add x17,x4,x7     // x17 is tmp result
    add x12,x10,x9
    LSL x16,x15,#1   // 4 row 
	
first_row_3_loop_8:
    sub x14,x2,#17 
    cmp x14,x13 
    blt first_row_3_loop_4
	
	 //input line 0 
    ld2 {v0.4s,v1.4s},[x10],#32 
    ld2 {v2.4s,v3.4s},[x10],#32  
    ldr s4,[x10]
    ext v5.16b, v0.16b,v2.16b,   #4
    fmul v14.4s, v0.4s,v24.s[0]
    fmul v15.4s, v2.4s,v24.s[0]	
    ext v13.16b,v2.16b,v4.16b,   #4
    fmla v14.4s, v1.4s,v24.s[1]
    fmla v15.4s, v3.4s,v24.s[1]
    sub x12,x10,#64
    prfm  pldl1keep, [x12,x15]    // row 4_0
	
    fmla v14.4s, v5.4s,v24.s[2]
    fmla v15.4s, v13.4s,v24.s[2]
    add x12,x12,x9
	
    //input line 1
    ld2 {v6.4s,v7.4s},[x12],#32 
    ld2 {v8.4s,v9.4s},[x12],#32  
    ldr s10,[x12]
	
    ext v27.16b,v6.16b,v8.16b,   #4
    ext v28.16b,v8.16b,v10.16b,  #4

    fmla v14.4s, v6.4s,v25.s[0]
    fmla v15.4s, v8.4s,v25.s[0]
    sub x12,x12,#64
	
    fmla v14.4s, v7.4s,v25.s[1]
    fmla v15.4s, v9.4s,v25.s[1]
    prfm  pldl1keep, [x12,x15]    // row 4_1
	
    fmla v14.4s, v27.4s,v25.s[2]
    fmla v15.4s, v28.4s,v25.s[2]
    add x12,x12,x9
	
    //input line 2
    ld2 {v0.4s,v1.4s},[x12],#32 
    ld2 {v2.4s,v3.4s},[x12],#32  
    ldr s4,[x12]
	
    ext v5.16b, v0.16b,v2.16b,   #4
	
	fmla v14.4s, v0.4s,v26.s[0]
    fmla v15.4s, v2.4s,v26.s[0]
	
    fmul v16.4s, v0.4s,v24.s[0]
    fmul v17.4s, v2.4s,v24.s[0]
    sub x12,x12,#64
	
    ext v13.16b,v2.16b,v4.16b,   #4
    fmla v14.4s, v1.4s,v26.s[1]
    fmla v15.4s, v3.4s,v26.s[1]
    fmla v16.4s, v1.4s,v24.s[1]
    fmla v17.4s, v3.4s,v24.s[1]
	prfm  pldl1keep, [x12,x15]   // row 4_2
	fmla v14.4s, v5.4s,v26.s[2]
	fmla v15.4s, v13.4s,v26.s[2]
    fmla v16.4s, v5.4s,v24.s[2]
	fmla v17.4s, v13.4s,v24.s[2]
	
	//add bias   
    fadd v14.4s,v14.4s,v21.4s  
    fadd v15.4s,v15.4s,v21.4s    
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:

#endif
    stp q14,q15,[x8]
	add x17,x8,x7	
    stp q16,q17,[x17]
	add x8,x8,#32
	add x13,x13,#16
    b first_row_3_loop_8
first_row_3_loop_4:
    sub x14,x2,#9
    cmp x14,x13
	blt first_row_3_loop_2
	
    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
    ldr s2,[x10]
	
	ext v5.16b, v0.16b,v2.16b,   #4
	sub x12,x10,#32
    fmul v14.4s, v0.4s,v24.s[0]
	prfm  pldl1keep, [x12,x15]   // row 4_0
    fmla v14.4s, v1.4s,v24.s[1]
	add x12,x12,x9
    fmla v14.4s, v5.4s,v24.s[2]
		
	//line 1
	ld2 {v6.4s,v7.4s},[x12],#32  
    ldr s8,[x12]
 
    ext v27.16b,v6.16b,v8.16b,   #4
	fmla v14.4s, v6.4s,v25.s[0]
	sub x12,x12,#32
	fmla v14.4s, v7.4s,v25.s[1]
	prfm  pldl1keep, [x12,x15] // row 4_1
	fmla v14.4s, v27.4s,v25.s[2]
	add x12,x12,x9
	
	//line 2
	ld2 {v0.4s,v1.4s},[x12],#32 
    ldr s2,[x12]
	ext v5.16b, v0.16b,v2.16b,   #4
	
	fmla v14.4s, v0.4s,v26.s[0]
    fmul v15.4s, v0.4s,v24.s[0]
	sub x12,x12,#32
	fmla v14.4s, v1.4s,v26.s[1]
    fmla v15.4s, v1.4s,v24.s[1]
	prfm  pldl1keep, [x12,x15]   // row 4_2
	fmla v14.4s, v5.4s,v26.s[2]
    fmla v15.4s, v5.4s,v24.s[2]
	
//add bias   
    fadd v14.4s,v14.4s,v21.4s  
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif    
    str q14,[x8]
	add x17,x8,x7
    str q15,[x17]
	add x8,x8,#16
	add x13,x13,#8
 
     b first_row_3_loop_4

first_row_3_loop_2:
	sub x14,x2,#5
    cmp x14,x13
	blt first_row_3_last_4_3_2
	
    movi d1,#0
    movi d7,#0
 
    //line 0 
    ldr q0,[x10],#16
    ldr s1,[x10]
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0
    ext   v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
	
	fmul v14.4s, v2.4s,v24.s[0]
	sub x12,x10,#16
    fmla v14.4s, v3.4s,v24.s[1]
	add x12,x12,x9
    fmla v14.4s, v4.4s,v24.s[2]
	
	//line 1
    ldr q6,[x12],#16
    ldr s7,[x12]
      
    uzp1  v8.4s,v6.4s,v7.4s        //a  c  e 0
    uzp2  v9.4s,v6.4s,v7.4s        //b  d  0 0
    ext   v10.16b,v8.16b,v31.16b,#4 //c  e  0 0
   
	fmla v14.4s, v8.4s,v25.s[0]
	sub x12,x12,#16
    fmla v14.4s, v9.4s,v25.s[1]
	add x12,x12,x9
    fmla v14.4s, v10.4s,v25.s[2]
	
	//line 2
	ldr q0,[x12],#16
    ldr s1,[x12]
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0

	fmla v14.4s, v2.4s,v26.s[0]
	fmul v15.4s, v2.4s,v24.s[0]
	sub x12,x12,#16
	ext  v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
	fmla v14.4s, v3.4s,v26.s[1]
    fmla v15.4s, v3.4s,v24.s[1]
	fmla v14.4s, v4.4s,v26.s[2]
    fmla v15.4s, v4.4s,v24.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s 	
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif    
    str d14,[x8]
	add x17,x8,x7
    str d15,[x17]
	add x8,x8,#8
	add x13,x13,#4
    b first_row_3_loop_2

first_row_3_last_4_3_2:    
    sub x14,x2,x13
    cmp x14,#4
	beq first_row_3_last_4
	cmp x14,#3
	beq first_row_3_last_3
	cmp x14,#2
	beq first_row_3_last_2
	cmp x14,#2
	blt first_row_3_end

first_row_3_last_4: 
  
   //line 0
   ldr q0,[x10] 
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0

   fmul v14.4s, v2.4s,v24.s[0]
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   fmla v14.4s, v3.4s,v24.s[1]
   add x12,x10,x9
   fmla v14.4s, v4.4s,v24.s[2]
   
   ldr q6,[x12] 
   
   uzp1  v8.4s,v6.4s,v31.4s        //a  c  0 0
   uzp2  v9.4s,v6.4s,v31.4s        //b  d  0 0
   ext   v10.16b,v8.16b,v31.16b,#4  //c  0  0 0 
   
   fmla v14.4s, v8.4s,v25.s[0]
   fmla v14.4s, v9.4s,v25.s[1]
   fmla v14.4s, v10.4s,v25.s[2]
   
   //line 2
   add x12,x12,x9
   ldr q0,[x12] 
 
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   
   fmla v14.4s, v2.4s,v26.s[0]
   fmul v15.4s, v2.4s,v24.s[0]
   fmla v14.4s, v3.4s,v26.s[1]
   fmla v15.4s, v3.4s,v24.s[1]
   fmla v14.4s, v4.4s,v26.s[2]
   fmla v15.4s, v4.4s,v24.s[2]
   
//add bias
   fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif   
   str d14,[x8]
   add x17,x8,x7
   str d15,[x17]

   b first_row_3_end
   
first_row_3_last_3:
   add x12,x10,x9
   add x14,x12,x9
   
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]
   ldp s6,s7,[x12]
   ldr s8,[x12,#8]
 
   fmul s14, s0,v24.s[0]
   fmla s14, s6,v25.s[0]
   fmla s14, s1,v24.s[1]
   fmla s14, s7,v25.s[1]
   fmla s14, s2,v24.s[2] 
   fmla s14, s8,v25.s[2] 
  
   ldp s0,s1,[x14]
   ldr s2,[x14,#8]  
 
   fmla s14, s0,v26.s[0]
   fmul s15, s0,v24.s[0]  
   add x12,x14,x9
   fmla s14, s1,v26.s[1]
   fmla s15, s1,v24.s[1]
   fmla s14, s2,v26.s[2]
   fmla s15, s2,v24.s[2] 
//add bias   
    fadd s14,s14,s21 

#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:

#endif
   str s14,[x8]
   add x17,x8,x7
   str s15,[x17]

   b first_row_3_end
   
first_row_3_last_2:
  
   //line 0
   ldp s0,s1,[x10]  
   fmul s14, s0,v24.s[0]
   fmla s14, s1,v24.s[1] 
   
   //line 1
   add x12,x10,x9
   ldp s6,s7,[x12]
   fmla s14, s6,v25.s[0]
   fmla s14, s7,v25.s[1]

   //line 2
   add x12,x12,x9
   ldp s0,s1,[x12]
   
   fmla s14, s0,v26.s[0]
   fmla s14, s1,v26.s[1]
   fmul s15, s0,v24.s[0]
   fmla s15, s1,v24.s[1]
 
//add bias   
    fadd s14,s14,s21 
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:

#endif
   str s14,[x8]
   add x17,x8,x7 
   str s15,[x17]
   
first_row_3_end:
   add x4,x4,x7           //output add one rows
   mov x8,x4 
   add x11,x11,#3         //input add three rows

   add x0,x0,x9
   add x0,x0,x9, LSL #1   //input add three rows 
   mov x10,x0
   mov x13,#0
   
mid_row_4_loop:
   subs x14,x1,x11
   cmp x14,#4
   ble mid_row_2_loop
 	
mid_row_4_loop_8: 
    sub x14,x2,#17 
    cmp x14,x13 
    blt mid_row_4_loop_4
	
    ldr q14,[x8]
    ldr q15,[x8,#16]
	
    //input line 0 
    ld2 {v0.4s,v1.4s},[x10],#32 
    ld2 {v2.4s,v3.4s},[x10],#32  
    ldr s4,[x10]
    ext v5.16b, v0.16b,v2.16b,   #4
    fmla v14.4s, v0.4s,v25.s[0]
    fmla v15.4s, v2.4s,v25.s[0]	
    ext v13.16b,v2.16b,v4.16b,   #4
    fmla v14.4s, v1.4s,v25.s[1]
    fmla v15.4s, v3.4s,v25.s[1]
    sub x12,x10,#64
    prfm  pldl1keep, [x12,x15]    // row 4_0
	
    fmla v14.4s, v5.4s,v25.s[2]
    fmla v15.4s, v13.4s,v25.s[2]
    add x12,x12,x9
	
    //input line 1
    ld2 {v6.4s,v7.4s},[x12],#32 
    ld2 {v8.4s,v9.4s},[x12],#32  
    ldr s10,[x12]
	
    ext v27.16b,v6.16b,v8.16b,   #4
    ext v28.16b,v8.16b,v10.16b,  #4

    fmla v14.4s, v6.4s,v26.s[0]
    fmla v15.4s, v8.4s,v26.s[0]
    fmul v16.4s, v6.4s,v24.s[0]
    fmul v17.4s, v8.4s,v24.s[0]
    sub x12,x12,#64
	
    fmla v14.4s, v7.4s,v26.s[1]
    fmla v15.4s, v9.4s,v26.s[1]
    fmla v16.4s, v7.4s,v24.s[1]	
    fmla v17.4s, v9.4s,v24.s[1]
    prfm  pldl1keep, [x12,x15]    // row 4_1
	
    fmla v14.4s, v27.4s,v26.s[2]
    fmla v15.4s, v28.4s,v26.s[2]
    fmla v16.4s, v27.4s,v24.s[2]
    fmla v17.4s, v28.4s,v24.s[2]
    add x12,x12,x9
	
    //input line 2
    ld2 {v0.4s,v1.4s},[x12],#32 
    ld2 {v2.4s,v3.4s},[x12],#32  
    ldr s4,[x12]
	
    ext v5.16b, v0.16b,v2.16b,   #4
    fmla v16.4s, v0.4s,v25.s[0]
    fmla v17.4s, v2.4s,v25.s[0]
    sub x12,x12,#64
	
    ext v13.16b,v2.16b,v4.16b,   #4
    fmla v16.4s, v1.4s,v25.s[1]
    fmla v17.4s, v3.4s,v25.s[1]
	prfm  pldl1keep, [x12,x15]   // row 4_2
	
    fmla v16.4s, v5.4s,v25.s[2]
	fmla v17.4s, v13.4s,v25.s[2]
	add x12,x12,x9

    //input line 3
	ld2 {v6.4s,v7.4s},[x12],#32 
	ld2 {v8.4s,v9.4s},[x12],#32  
    ldr s10,[x12]
	
	fmla v16.4s, v6.4s,v26.s[0]
	fmla v17.4s, v8.4s,v26.s[0]
	fmul v18.4s, v6.4s,v24.s[0]
	fmul v19.4s, v8.4s,v24.s[0]
	
	ext v27.16b,v6.16b,v8.16b,   #4
    fmla v16.4s, v7.4s,v26.s[1]
	fmla v17.4s, v9.4s,v26.s[1]
	
	ext v28.16b,v8.16b,v10.16b,  #4
    fmla v18.4s, v7.4s,v24.s[1]
	fmla v19.4s, v9.4s,v24.s[1]
	
    fmla v16.4s, v27.4s,v26.s[2]
	fmla v17.4s, v28.4s,v26.s[2]
	fmla v18.4s, v27.4s,v24.s[2]
	fmla v19.4s, v28.4s,v24.s[2]
	
	sub x12,x12,#64
	prfm  pldl1keep, [x12,x15]  // row 4_3
	
//add bias   
    fadd v14.4s,v14.4s,v21.4s  
    fadd v15.4s,v15.4s,v21.4s   
	fadd v16.4s,v16.4s,v21.4s  
    fadd v17.4s,v17.4s,v21.4s  
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    fmax v16.4s,v16.4s,v31.4s
    fmax v17.4s,v17.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
    fmin v16.4s,v16.4s,v29.4s
    fmin v17.4s,v17.4s,v29.4s
100:    

#endif
    stp q14,q15,[x8]
	add x17,x8,x7	
    stp q16,q17,[x17]
	add x17,x17,x7
	add x8,x8,#32
	add x13,x13,#16
	stp q18,q19,[x17]

	b  mid_row_4_loop_8

mid_row_4_loop_4:   
	sub x14,x2,#9
    cmp x14,x13
	blt mid_row_4_loop_2
	
	ldr q14,[x8]
	
    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
    ldr s2,[x10]
	
	ext v5.16b, v0.16b,v2.16b,   #4
	sub x12,x10,#32
    fmla v14.4s, v0.4s,v25.s[0]
	prfm  pldl1keep, [x12,x15]   // row 4_0
    fmla v14.4s, v1.4s,v25.s[1]
	add x12,x12,x9
    fmla v14.4s, v5.4s,v25.s[2]
		
	//line 1
	ld2 {v6.4s,v7.4s},[x12],#32  
    ldr s8,[x12]
 
    ext v27.16b,v6.16b,v8.16b,   #4
	fmul v15.4s, v6.4s,v24.s[0]
	fmla v14.4s, v6.4s,v26.s[0]
	sub x12,x12,#32
    fmla v15.4s, v7.4s,v24.s[1]
	fmla v14.4s, v7.4s,v26.s[1]
	prfm  pldl1keep, [x12,x15] // row 4_1
	
    fmla v15.4s, v27.4s,v24.s[2]
	fmla v14.4s, v27.4s,v26.s[2]
	add x12,x12,x9
	
	//line 2
	ld2 {v0.4s,v1.4s},[x12],#32 
    ldr s2,[x12]
	ext v5.16b, v0.16b,v2.16b,   #4
	
    fmla v15.4s, v0.4s,v25.s[0]
	sub x12,x12,#32
    fmla v15.4s, v1.4s,v25.s[1]
	prfm  pldl1keep, [x12,x15]   // row 4_2
    fmla v15.4s, v5.4s,v25.s[2]
	add x12,x12,x9
	
	//line 3
	ld2 {v6.4s,v7.4s},[x12],#32  
    ldr s8,[x12]
    
    ext v27.16b,v6.16b,v8.16b,   #4
	fmul v16.4s, v6.4s,v24.s[0]
	fmla v15.4s, v6.4s,v26.s[0]
	sub x12,x12,#32
	
    fmla v16.4s, v7.4s,v24.s[1]
	fmla v15.4s, v7.4s,v26.s[1]
	prfm  pldl1keep, [x12,x15]  // row 4_3
	
    fmla v16.4s, v27.4s,v24.s[2]
	fmla v15.4s, v27.4s,v26.s[2]
	
//add bias   
    fadd v14.4s,v14.4s,v21.4s  
    fadd v15.4s,v15.4s,v21.4s	
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:
#endif    
    str q14,[x8]
	add x17,x8,x7
    str q15,[x17]
	add x17,x17,x7
	add x8,x8,#16
	add x13,x13,#8
    str q16,[x17]
	
	b mid_row_4_loop_4

mid_row_4_loop_2: 
    sub x14,x2,#5
    cmp x14,x13
	blt mid_row_4_last_4_3_2
	
	ldr d14,[x8] 

    movi d1,#0
    movi d7,#0
 
    //line 0 
    ldr q0,[x10],#16
    ldr s1,[x10]
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0
    ext   v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
	
	fmla v14.4s, v2.4s,v25.s[0]
	sub x12,x10,#16
    fmla v14.4s, v3.4s,v25.s[1]
	add x12,x12,x9
    fmla v14.4s, v4.4s,v25.s[2]
	
	//line 1
    ldr q6,[x12],#16
    ldr s7,[x12]
      
    uzp1  v8.4s,v6.4s,v7.4s        //a  c  e 0
    uzp2  v9.4s,v6.4s,v7.4s        //b  d  0 0
    ext   v10.16b,v8.16b,v31.16b,#4 //c  e  0 0
   
    fmul v15.4s, v8.4s,v24.s[0]
	fmla v14.4s, v8.4s,v26.s[0]
	sub x12,x12,#16
    fmla v15.4s, v9.4s,v24.s[1]
    fmla v14.4s, v9.4s,v26.s[1]
	add x12,x12,x9
    fmla v15.4s, v10.4s,v24.s[2]
    fmla v14.4s, v10.4s,v26.s[2]
	
	//line 2
	ldr q0,[x12],#16
    ldr s1,[x12]
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0

	fmla v15.4s, v2.4s,v25.s[0]
	sub x12,x12,#16
	ext  v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
	
    fmla v15.4s, v3.4s,v25.s[1]
	add x12,x12,x9
    fmla v15.4s, v4.4s,v25.s[2]
	
	//line 3
    ldr q6,[x12],#16
    ldr s7,[x12]
      
    uzp1  v8.4s,v6.4s,v7.4s        //a  c  e 0
    uzp2  v9.4s,v6.4s,v7.4s        //b  d  0 0
  
    fmla v15.4s, v8.4s,v26.s[0]
    fmul v16.4s, v8.4s,v24.s[0]
	fadd v14.4s,v14.4s,v21.4s
	
	ext  v10.16b,v8.16b,v31.16b,#4 //c  e  0 0
    fmla v15.4s, v9.4s,v26.s[1] 
    fmla v16.4s, v9.4s,v24.s[1]
		
    fmla v15.4s, v10.4s,v26.s[2]
    fmla v16.4s, v10.4s,v24.s[2]
    fadd v15.4s,v15.4s,v21.4s 	
	
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:
#endif    
    str d14,[x8]
	add x17,x8,x7
    str d15,[x17]
	add x17,x17,x7
	add x8,x8,#8
	add x13,x13,#4
    str d16,[x17]
	
	b mid_row_4_loop_2

mid_row_4_last_4_3_2:    
    sub x14,x2,x13
    cmp x14,#4
	beq mid_row_4_last_4
	cmp x14,#3
	beq mid_row_4_last_3
	cmp x14,#2
	beq mid_row_4_last_2
	cmp x14,#2
	blt mid_row_4_end

mid_row_4_last_4: 

   ldr d14,[x8] 
   //line 0
   ldr q0,[x10] 
 
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0

   fmla v14.4s, v2.4s,v25.s[0]
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   fmla v14.4s, v3.4s,v25.s[1]
   add x12,x10,x9
   fmla v14.4s, v4.4s,v25.s[2]
   
   ldr q6,[x12] 
   
   uzp1  v8.4s,v6.4s,v31.4s        //a  c  0 0
   uzp2  v9.4s,v6.4s,v31.4s        //b  d  0 0
   ext   v10.16b,v8.16b,v31.16b,#4  //c  0  0 0 
   
   fmla v14.4s, v8.4s,v26.s[0]
   fmul v15.4s, v8.4s,v24.s[0]
   fmla v14.4s, v9.4s,v26.s[1]
   fmla v15.4s, v9.4s,v24.s[1]
   fmla v14.4s, v10.4s,v26.s[2]
   fmla v15.4s, v10.4s,v24.s[2]
  
   //line 2
   add x12,x12,x9
   ldr q0,[x12] 
 
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   
   fmla v15.4s, v2.4s,v25.s[0]
   add x12,x12,x9
   fmla v15.4s, v3.4s,v25.s[1]
   fmla v15.4s, v4.4s,v25.s[2]
   
   ldr q6,[x12] 
   
   uzp1  v8.4s,v6.4s,v31.4s        //a  c  0 0
   uzp2  v9.4s,v6.4s,v31.4s        //b  d  0 0
  
   
   fmla v15.4s, v8.4s,v26.s[0]
   fmul v16.4s, v8.4s,v24.s[0]
   ext   v10.16b,v8.16b,v31.16b,#4  //c  0  0 0 
   fmla v15.4s, v9.4s,v26.s[1]
   fmla v16.4s, v9.4s,v24.s[1]
   fadd v14.4s,v14.4s,v21.4s
   fmla v15.4s, v10.4s,v26.s[2]
   fmla v16.4s, v10.4s,v24.s[2]
  
   fadd v15.4s,v15.4s,v21.4s 	
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt  100f
   fmax v14.4s,v14.4s,v31.4s
   fmax v15.4s,v15.4s,v31.4s
   beq 100f
   fmin v14.4s,v14.4s,v29.4s
   fmin v15.4s,v15.4s,v29.4s
100:
#endif   
   str d14,[x8]
   add x17,x8,x7
   str d15,[x17]
   add x17,x17,x7
   str d16,[x17]
   
   b mid_row_4_end
   
mid_row_4_last_3:  
 
   add x12,x10,x9
   add x14,x12,x9
   
   ldr s14,[x8]   
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]
   ldp s6,s7,[x12]
   ldr s8,[x12,#8]
 
   fmla s14, s0,v25.s[0]
   fmul s15, s6,v24.s[0]
   fmla s14, s1,v25.s[1]
   fmla s15, s7,v24.s[1]
   fmla s14, s2,v25.s[2] 
   fmla s15, s8,v24.s[2] 
  
   ldp s0,s1,[x14]
   ldr s2,[x14,#8]  
 
   fmla s14, s6,v26.s[0]
   fmla s15, s0,v25.s[0]  
   add x12,x14,x9
   fmla s14, s7,v26.s[1]
   fmla s15, s1,v25.s[1]
   fmla s14, s8,v26.s[2]
   fmla s15, s2,v25.s[2] 

   ldp s6,s7,[x12]
   ldr s8,[x12,#8]  
   
   fmla s15, s6,v26.s[0]
   fmul s16, s6,v24.s[0]
   fadd s14,s14,s21 
   fmla s15, s7,v26.s[1]
   fmla s16, s7,v24.s[1]
   fmla s15, s8,v26.s[2] 
   fmla s16, s8,v24.s[2]
   fadd s15,s15,s21 	
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   fmax s15,s15,s31
   beq 100f
   fmin s14,s14,s29
   fmin s15,s15,s29
100:
#endif
   str s14,[x8]
   add x17,x8,x7
   str s15,[x17]
   add x17,x17,x7
   str s16,[x17]

   b mid_row_4_end
   
mid_row_4_last_2:
   ldr s14,[x8] 
 
   //line 0
   ldp s0,s1,[x10]  
   fmla s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1] 
   
   //line 1
   add x12,x10,x9
   ldp s6,s7,[x12]
 
   fmul s15, s6,v24.s[0]
   fmla s14, s6,v26.s[0]
   fmla s15, s7,v24.s[1]
   fmla s14, s7,v26.s[1]
 
   //line 2
   add x12,x12,x9
   ldp s0,s1,[x12]
   
   fmla s15, s0,v25.s[0]
   fmla s15, s1,v25.s[1]
   
   //line 3
   add x12,x12,x9
   ldp s6,s7,[x12]
 
   fmul s16, s6,v24.s[0]
   fmla s15, s6,v26.s[0]
   fmla s16, s7,v24.s[1]
   fmla s15, s7,v26.s[1]
 
//add bias   
    fadd s14,s14,s21 
    fadd s15,s15,s21	
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   fmax s15,s15,s31
   beq 100f
   fmin s14,s14,s29
   fmin s15,s15,s29
100:
#endif
   str s14,[x8]
   add x17,x8,x7 
   str s15,[x17]
   add x17,x17,x7
   str s16,[x17]
mid_row_4_end:
   add x4,x4,x7,LSL #1   //output add two rows
   mov x8,x4 
   add x11,x11,#4        //input add four rows
   add x0,x0,x9,LSL #2
   mov x10,x0
   mov x13,#0
   b mid_row_4_loop
   
mid_row_2_loop:
   add x12,x10,x9
   add x17,x8,x7
   
   subs x14,x1,x11
   cmp x14,#2
   beq last_row_2_loop
   cmp x14,#1
   beq last_row_loop
   	
mid_row_2_loop_8: 
    sub x14,x2,#17 
    cmp x14,x13
	blt mid_row_2_loop_4

	ldr q14,[x8]
    ldr q15,[x8,#16]
	ld2 {v0.4s,v1.4s},[x10],#32 
	ld2 {v2.4s,v3.4s},[x10],#32  
    ldr s4,[x10]
	ld2 {v6.4s,v7.4s},[x12],#32 
	ld2 {v8.4s,v9.4s},[x12],#32  
    ldr s10,[x12]
	
	ext v5.16b, v0.16b,v2.16b,   #4
    ext v13.16b,v2.16b,v4.16b,   #4

 
    fmla v14.4s, v0.4s,v25.s[0]
	fmla v15.4s, v2.4s,v25.s[0]
	fmul v16.4s, v6.4s,v24.s[0]
	fmul v17.4s, v8.4s,v24.s[0]
	ext v27.16b,v6.16b,v8.16b,   #4
	
    fmla v14.4s, v1.4s,v25.s[1]
	fmla v15.4s, v3.4s,v25.s[1]
	fmla v16.4s, v7.4s,v24.s[1]
	fmla v17.4s, v9.4s,v24.s[1]
	ext v28.16b,v8.16b,v10.16b,  #4
	
    fmla v14.4s, v5.4s,v25.s[2]
    fmla v15.4s, v13.4s,v25.s[2]
    fmla v16.4s, v27.4s,v24.s[2]
    fmla v17.4s, v28.4s,v24.s[2]
		
    fmla v14.4s, v6.4s,v26.s[0]
	fmla v15.4s, v8.4s,v26.s[0]
    fmla v14.4s, v7.4s,v26.s[1]
	fmla v15.4s, v9.4s,v26.s[1]
    fmla v14.4s, v27.4s,v26.s[2]
    fmla v15.4s, v28.4s,v26.s[2]
	
	add x13,x13,#16
//add bias   
    fadd v14.4s,v14.4s,v21.4s  
    fadd v15.4s,v15.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:
#endif
    stp q14,q15,[x8],#32
    stp q16,q17,[x17],#32
    
	b  mid_row_2_loop_8
    
mid_row_2_loop_4: 
    sub x14,x2,#9
    cmp x14,x13
	blt mid_row_2_loop_2

	ldr q14,[x8]
	
    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
    ldr s2,[x10]
	ld2 {v6.4s,v7.4s},[x12],#32  
    ldr s8,[x12]
    
	ext v5.16b, v0.16b,v2.16b,   #4
    fmla v14.4s, v0.4s,v25.s[0]
	fmul v16.4s, v6.4s,v24.s[0]
	
	ext v27.16b,v6.16b,v8.16b,   #4
    fmla v14.4s, v1.4s,v25.s[1]
	fmla v16.4s, v7.4s,v24.s[1]
	
    fmla v14.4s, v5.4s,v25.s[2]
    fmla v16.4s, v27.4s,v24.s[2]
	
    fmla v14.4s, v6.4s,v26.s[0]	
	add x13,x13,#8
    fmla v14.4s, v7.4s,v26.s[1]
    fmla v14.4s, v27.4s,v26.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif    
    str q14,[x8],#16
    str q16,[x17],#16

	b mid_row_2_loop_4

mid_row_2_loop_2: 
    sub x14,x2,#5
    cmp x14,x13
	blt mid_row_2_last_4_3_2
    
	ldr d14,[x8] 
    movi d1,#0
    movi d7,#0
 
    ldr q0,[x10],#16
    ldr s1,[x10]
    ldr q6,[x12],#16
    ldr s7,[x12]
     
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0
    ext   v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
   
    uzp1  v8.4s,v6.4s,v7.4s        //a  c  e 0
    fmla v14.4s, v2.4s,v25.s[0]
    uzp2  v9.4s,v6.4s,v7.4s        //b  d  0 0
    fmla v14.4s, v3.4s,v25.s[1]
    ext   v10.16b,v8.16b,v31.16b,#4 //c  e  0 0
    fmla v14.4s, v4.4s,v25.s[2]
   
   	fmul v16.4s, v8.4s,v24.s[0]
    fmla v14.4s, v8.4s,v26.s[0]
    fmla v16.4s, v9.4s,v24.s[1]
    fmla v14.4s, v9.4s,v26.s[1]
    fmla v16.4s, v10.4s,v24.s[2]
    fmla v14.4s, v10.4s,v26.s[2]
	
    add x13,x13,#4
 //add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif    
    str d14,[x8],#8
    str d16,[x17],#8

	b mid_row_2_loop_2

mid_row_2_last_4_3_2:    
    sub x14,x2,x13
    cmp x14,#4
	beq mid_row_2_last_4
	cmp x14,#3
	beq mid_row_2_last_3
	cmp x14,#2
	beq mid_row_2_last_2
	cmp x14,#2
	blt mid_row_2_end

mid_row_2_last_4: 
   ldr q0,[x10] 
   ldr q6,[x12] 
   ldr d14,[x8] 
   
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   
   uzp1  v8.4s,v6.4s,v31.4s        //a  c  0 0   
   fmla v14.4s, v2.4s,v25.s[0]
   uzp2  v9.4s,v6.4s,v31.4s        //b  d  0 0
   fmla v14.4s, v3.4s,v25.s[1]
   ext   v10.16b,v8.16b,v31.16b,#4  //c  0  0 0 
   fmla v14.4s, v4.4s,v25.s[2]
   
   fmul v15.4s, v8.4s,v24.s[0]
   fmla v14.4s, v8.4s,v26.s[0]
   fmla v15.4s, v9.4s,v24.s[1]
   fmla v14.4s, v9.4s,v26.s[1]
   fmla v15.4s, v10.4s,v24.s[2]
   fmla v14.4s, v10.4s,v26.s[2]
   
 //add bias   
   fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax v14.4s,v14.4s,v31.4s
   beq 100f
   fmin v14.4s,v14.4s,v29.4s
100:

#endif   
   str d14,[x8]
   str d15,[x17]
   
   b mid_row_2_end
   
mid_row_2_last_3:  
   
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]    
   ldp s6,s7,[x12]
   ldr s8,[x12,#8]
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmul s15, s6,v24.s[0]
   fmla s14, s1,v25.s[1]
   fmla s15, s7,v24.s[1]
   fmla s14, s2,v25.s[2]
   fmla s15, s8,v24.s[2]
   
   fmla s14, s6,v26.s[0]
   fmla s14, s7,v26.s[1]
   fmla s14, s8,v26.s[2]
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:
#endif
   str s14,[x8]
   str s15,[x17]

   b mid_row_2_end
mid_row_2_last_2:
 
   ldp s0,s1,[x10]  
   ldp s6,s7,[x12]
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmul s15, s6,v24.s[0]
   fmla s14, s1,v25.s[1] 
   fmla s15, s7,v24.s[1]
   fmla s14, s6,v26.s[0]
   fmla s14, s7,v26.s[1]
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:
#endif
   str s14,[x8]
   str s15,[x17]
mid_row_2_end:
   add x4,x4,x7
   mov x8,x4
   add x17,x4,x7
   add x11,x11,#2
   add x0,x0,x9,LSL #1
   mov x10,x0
   add x12,x0,x9
   mov x13,#0

   b mid_row_2_loop
   
last_row_2_loop:
   mov x10,x0
   add x12,x0,x9
   mov  x15,#8
   mul  x15,x2,x15
   
last_row_2_loop_8: 
    sub x14,x2,#17 
    cmp x14,x13
	blt last_row_2_loop_4

    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
	ld2 {v2.4s,v3.4s},[x10],#32  
    ldr s4,[x10]
	
	ld2 {v6.4s,v7.4s},[x12],#32 
	ld2 {v8.4s,v9.4s},[x12],#32  
    ldr s10,[x12]
	
	ldr q14,[x8]
    ldr q15,[x8,#16]
	
	ext v5.16b, v0.16b,v2.16b,   #4
    fmla v14.4s, v0.4s,v25.s[0]
	fmla v15.4s, v2.4s,v25.s[0]
	ext v13.16b,v2.16b,v4.16b,   #4
    fmla v14.4s, v1.4s,v25.s[1]
	fmla v15.4s, v3.4s,v25.s[1]
    ext v27.16b,v6.16b,v8.16b,   #4
    fmla v14.4s, v5.4s,v25.s[2]
    fmla v15.4s, v13.4s,v25.s[2]
    ext v28.16b,v8.16b,v10.16b,  #4
    fmla v14.4s, v6.4s,v26.s[0]
	fmla v15.4s, v8.4s,v26.s[0]
	sub x16,x10,#64
	prfm  pldl1keep, [x16,x15]   
    fmla v14.4s, v7.4s,v26.s[1]
    fmla v15.4s, v9.4s,v26.s[1]
    sub x16,x12,#64
	prfm  pldl1keep, [x16,x15]  
    fmla v14.4s, v27.4s,v26.s[2]
    fmla v15.4s, v28.4s,v26.s[2]
	add x13,x13,#16
 //add bias   
    fadd v14.4s,v14.4s,v21.4s  
    fadd v15.4s,v15.4s,v21.4s  	
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:
#endif 
    stp q14,q15,[x8],#32
 
	b  last_row_2_loop_8

last_row_2_loop_4:
    sub x14,x2,#9
    cmp x14,x13
	blt last_row_2_loop_2

    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
    ldr s2,[x10]
	ld2 {v6.4s,v7.4s},[x12],#32  
    ldr s8,[x12] 
	ldr q14,[x8]
	
	ext v5.16b, v0.16b,v2.16b,   #4

    fmla v14.4s, v0.4s,v25.s[0]
    fmla v14.4s, v1.4s,v25.s[1]
    fmla v14.4s, v5.4s,v25.s[2]
    
	ext v27.16b,v6.16b,v8.16b,   #4
    fmla v14.4s, v6.4s,v26.s[0]
    fmla v14.4s, v7.4s,v26.s[1]
    fmla v14.4s, v27.4s,v26.s[2]
	add x13,x13,#8
	
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif    
    str q14,[x8],#16
    
	b  last_row_2_loop_4
last_row_2_loop_2: 
  
    sub x14,x2,#5
    cmp x14,x13
	blt last_row_2_last_4_3_2

    movi d1,#0
    movi d7,#0
 
    ldr q0,[x10],#16
    ldr s1,[x10]
    ldr q6,[x12],#16
    ldr s7,[x12]
     
	ldr d14,[x8] 
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0
    ext   v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
   
    uzp1  v8.4s,v6.4s,v7.4s        //a  c  e 0
    fmla v14.4s,v2.4s,v25.s[0]
	uzp2  v9.4s,v6.4s,v7.4s        //b  d  0 0
    fmla v14.4s, v3.4s,v25.s[1]
	ext  v10.16b,v8.16b,v31.16b,#4 //c  e  0 0
    fmla v14.4s, v4.4s,v25.s[2]
   
    fmla v14.4s, v8.4s,v26.s[0]
    fmla v14.4s, v9.4s,v26.s[1]
    fmla v14.4s, v10.4s,v26.s[2]
	
    add x13,x13,#4
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif   
   
    str d14,[x8],#8
   
	b  last_row_2_loop_2

last_row_2_last_4_3_2:   
    sub x14,x2,x13  
    cmp x14,#4
	beq last_row_2_last_4
	cmp x14,#3
	beq last_row_2_last_3
	cmp x14,#2
	beq last_row_2_last_2
	cmp x14,#2
	blt last_row_end 

last_row_2_last_4: 
   ldr q0,[x10] 
   ldr q6,[x12] 
   ldr d14,[x8] 
   
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
   
   uzp1  v8.4s,v6.4s,v31.4s        //a  c  0 0
   fmla v14.4s, v2.4s,v25.s[0]
   uzp2  v9.4s,v6.4s,v31.4s        //b  d  0 0
   fmla v14.4s, v3.4s,v25.s[1]
   ext  v10.16b,v8.16b,v31.16b,#4  //c  0  0 0 
   fmla v14.4s, v4.4s,v25.s[2]
   
   fmla v14.4s, v8.4s,v26.s[0]
   fmla v14.4s, v9.4s,v26.s[1]
   fmla v14.4s, v10.4s,v26.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif     
   str d14,[x8]
   b last_row_end

last_row_2_last_3:  
   
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]   
   ldp s6,s7,[x12]
   ldr s8,[x12,#8]
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1]
   fmla s14, s2,v25.s[2]
   
   fmla s14, s6,v26.s[0]
   fmla s14, s7,v26.s[1]
   fmla s14, s8,v26.s[2]
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:
#endif
   str s14,[x8]
   b last_row_end
last_row_2_last_2:
 
   ldp s0,s1,[x10]  
   ldp s6,s7,[x12]
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1] 

   fmla s14, s6,v26.s[0]
   fmla s14, s7,v26.s[1]
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:
#endif
   str s14,[x8]
   
   b last_row_end
   
last_row_loop:
   mov x10,x0
   add x12,x0,x9
   mov  x15,#8
   mul  x15,x2,x15
	
last_row_loop_8: 

    sub x14,x2,#17 
    cmp x14,x13
	blt last_row_loop_4

      //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
	ld2 {v2.4s,v3.4s},[x10],#32  
    ldr s4,[x10]

	ldr q14,[x8]
    ldr q15,[x8,#16]
	
	ext v5.16b, v0.16b,v2.16b,   #4
    fmla v14.4s, v0.4s,v25.s[0]
	fmla v15.4s, v2.4s,v25.s[0]
	
	ext v13.16b,v2.16b,v4.16b,   #4
    fmla v14.4s, v1.4s,v25.s[1]
	fmla v15.4s, v3.4s,v25.s[1]
	
	sub x16,x10,#64
	prfm  pldl1keep, [x10,x15] 
	
    fmla v14.4s, v5.4s,v25.s[2]
    fmla v15.4s, v13.4s,v25.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
	fadd v15.4s,v15.4s,v21.4s 
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt  100f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
100:

#endif  
    stp q14,q15,[x8],#32
	add x13,x13,#16
	b last_row_loop_8

last_row_loop_4: 
    sub x14,x2,#9
    cmp x14,x13
	blt last_row_loop_2

    //line 0 
	ld2 {v0.4s,v1.4s},[x10],#32 
    ldr s2,[x10]
	
	ldr q14,[x8]
	
	ext v5.16b, v0.16b,v2.16b,   #4
   
    fmla v14.4s, v0.4s,v25.s[0]
    fmla v14.4s, v1.4s,v25.s[1]
    fmla v14.4s, v5.4s,v25.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif     		 
    str q14,[x8],#16
	
	add x13,x13,#8
	
	b last_row_loop_4
 
last_row_loop_2: 
    sub x14,x2,#5
    cmp x14,x13
	blt last_row_last_4_3_2

    movi d1,#0
    movi d7,#0
 
    ldr q0,[x10],#16
    ldr s1,[x10]

	ldr d14,[x8] 
	
    uzp1  v2.4s,v0.4s,v1.4s        //a  c  e 0
    uzp2  v3.4s,v0.4s,v1.4s        //b  d  0 0
    ext   v4.16b,v2.16b,v31.16b,#4 //c  e  0 0 
  
  
    fmla v14.4s, v2.4s,v25.s[0]
    fmla v14.4s, v3.4s,v25.s[1]
    fmla v14.4s, v4.4s,v25.s[2]
//add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif        
    st1 {v14.2s},[x8],#8
   
    add x13,x13,#4
	
	b last_row_loop_2

last_row_last_4_3_2: 
    sub x14,x2,x13  
    cmp x14,#4
	beq last_row_last_4
	cmp x14,#3
	beq last_row_last_3
	cmp x14,#2
	beq last_row_last_2
	cmp x14,#2
	blt last_row_end    

last_row_last_4: 
   ldr q0,[x10] 
   ldr d14,[x8] 
   
   uzp1  v2.4s,v0.4s,v31.4s        //a  c  0 0
   uzp2  v3.4s,v0.4s,v31.4s        //b  d  0 0
   ext   v4.16b,v2.16b,v31.16b,#4  //c  0  0 0 
    
   fmla v14.4s, v2.4s,v25.s[0]
   fmla v14.4s, v3.4s,v25.s[1]
   fmla v14.4s, v4.4s,v25.s[2]
 //add bias   
    fadd v14.4s,v14.4s,v21.4s   
#ifdef CONV_RELU_FUSE  
    cmp w18,0
    blt 100f
    fmax v14.4s,v14.4s,v31.4s
    beq 100f
    fmin v14.4s,v14.4s,v29.4s
100:

#endif       
   st1 {v14.2s},[x8]
   b last_row_end
last_row_last_3:  
   
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]
    
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1]
   fmla s14, s2,v25.s[2]
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:

#endif 
   str s14,[x8]
   b last_row_end
	
last_row_last_2:
 
   ldp s0,s1,[x10]  
   ldr s14,[x8] 
 
   fmla s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1] 
//add bias   
    fadd s14,s14,s21   
#ifdef CONV_RELU_FUSE  
   cmp w18,0
   blt 100f
   fmax s14,s14,s31
   beq 100f
   fmin s14,s14,s29
100:

#endif
   str s14,[x8]
last_row_end:
     ldp d8,d9,[sp]
     ldp d10,d11,[sp,0x10] 
     ldp d12,d13,[sp,0x20]
     ldp d14,d15,[sp,0x30]
     add sp,sp,#0x40

    ret

   

