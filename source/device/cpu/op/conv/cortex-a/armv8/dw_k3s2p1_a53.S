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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 */
//x0: input
//x1: h
//x2: w
//x3: kernel
//x4: output //L-2
//x5: bias

//x6: output point L0

//x7: output_w * sizeof(float)
//x8: output L1

//x9: width * sizeof(float)
//x10: input point tmp L0
//x11: input_h tmp
//x12:  input point tmp L1

//x13: input_w tmp

//x14: input_w tmp

//input: v0 ~ v4
//shift: v5,v13,v27,v28
//input: v6~v10


//kernel vector: v24 ~ v26

//output_vec  v14~v17

//bias: v21
//zero vector : v31

//v29:  saved activation

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p1_a53
#endif

#define CONV_RELU_FUSE

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:

   //save activation
   scvtf s29,w6
   dup v29.4s,v29.s[0]
   mov x18,x6

   //load kernel
   ldp q24,q25,[x3]
   ldr s26,[x3,#32]

   ext  v26.16b,v25.16b,v26.16b,8
   ext  v25.16b,v24.16b,v25.16b,12

   lsl x9,x2,2       //input_w * sizeof(float)

   add x7,x2,1       // x2 + 2pad - 3  pad_x = 1;kernel_size = 3
   lsr x7,x7,1       // x7 = x2/2 +1 ( stride = 2;)
   lsl x7,x7,2       // output_w * sizeof(float)

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
   b first_row_2_start

non_biases:
   movi d21,#0

first_row_2_start:
    movi d14,#0
	movi d15,#0
	mov x10,x0
	mov x8,x4
    add x6,x4,x7     // x6 is tmp result
	add x12,x10,x9

	LSL x16,x15,#1   // 4 row

first_row_2_first_4:

   ld2 {v0.4s,v1.4s},[x10]           //line 0
   ld2 {v2.4s,v3.4s},[x12]           //line 1

   ext v5.16b,v31.16b,v1.16b,  #12
   ext v27.16b,v31.16b,v3.16b, #12

   prfm	pldl1keep, [x10,x15]         // row 4_0

   fmul v14.4s, v5.4s,v25.s[0]
   fmul v15.4s, v27.4s,v24.s[0]
   prfm	pldl1keep, [x10,x16]         // row 4_2

   fmla v14.4s, v0.4s,v25.s[1]
   fmla v15.4s, v2.4s, v24.s[1]
   prfm	pldl1keep, [x12,x15]         // row 4_1

   fmla v14.4s, v1.4s,v25.s[2]
   fmla v15.4s, v3.4s, v24.s[2]

   fmla v14.4s, v27.4s,v26.s[0]
   prfm	pldl1keep, [x12,x16]         // row 4_3
   add x13,x13,#7

   fmla v14.4s, v2.4s, v26.s[1]
   add x10,x10,#28
   add x12,x12,#28
   fmla v14.4s, v3.4s, v26.s[2]

   //add bias
   fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_100f
   fmax v14.4s,v14.4s,v31.4s

save_done_100f:
#endif
   str q15,[x6],#16
   str q14,[x8],#16

first_row_2_loop_8:
    sub x14,x2,#17
    cmp x14,x13
	blt first_row_2_loop_4

    //line 0
	ld2 {v0.4s,v1.4s},[x10],#32
	ld2 {v2.4s,v3.4s},[x10],#32
    ldr s4,[x10]

	ld2 {v6.4s,v7.4s},[x12],#32
	ld2 {v8.4s,v9.4s},[x12],#32
    ldr s10,[x12]

	ext v5.16b, v0.16b,v2.16b,   #4
    fmul v14.4s, v0.4s,v25.s[0]
	fmul v15.4s, v2.4s,v25.s[0]

	ext v13.16b,v2.16b,v4.16b,   #4
	fmul v16.4s, v6.4s,v24.s[0]
	fmul v17.4s, v8.4s,v24.s[0]

	ext v27.16b,v6.16b,v8.16b,   #4
    fmla v14.4s, v1.4s,v25.s[1]
	fmla v15.4s, v3.4s,v25.s[1]

	ext v28.16b,v8.16b,v10.16b,  #4
	fmla v16.4s, v7.4s,v24.s[1]
	fmla v17.4s, v9.4s,v24.s[1]

    fmla v14.4s, v5.4s,v25.s[2]
	fmla v15.4s, v13.4s,v25.s[2]
	fmla v16.4s, v27.4s,v24.s[2]
	fmla v17.4s, v28.4s,v24.s[2]

    fmla v14.4s, v6.4s,v26.s[0]
	fmla v15.4s, v8.4s,v26.s[0]
	sub x16,x10,#64
    prfm  pldl1keep, [x16,x15]   // row 4_0

    fmla v14.4s, v7.4s,v26.s[1]
	fmla v15.4s, v9.4s,v26.s[1]
    add x16,x16,x9, LSL #1
    prfm  pldl1keep, [x16,x15]  // row 4_2

    fmla v14.4s, v27.4s,v26.s[2]
	fmla v15.4s, v28.4s,v26.s[2]
    sub x16,x12,#64
    prfm  pldl1keep, [x16,x15]  // row 4_1

//add bias
    fadd v14.4s,v14.4s,v21.4s
    fadd v15.4s,v15.4s,v21.4s

	add x16,x16,x9, LSL #1
    prfm  pldl1keep, [x16,x15]  // row 4_3
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_101f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s

save_done_101f:

#endif
    stp q14,q15,[x8],#32
    stp q16,q17,[x6],#32

	add x13,x13,#16
	b first_row_2_loop_8

first_row_2_loop_4:
    sub x14,x2,#9
    cmp x14,x13
	blt first_row_2_loop_2

	 //line 0
	ld2 {v0.4s,v1.4s},[x10],#32
    ldr s2,[x10]

	ld2 {v6.4s,v7.4s},[x12],#32
    ldr s8,[x12]

    fmul v14.4s, v0.4s,v25.s[0]
	ext v5.16b, v0.16b,v2.16b,   #4
	add x13,x13,#8
    fmla v14.4s, v1.4s,v25.s[1]
	ext v27.16b,v6.16b,v8.16b,   #4
    fmla v14.4s, v5.4s,v25.s[2]

	fmul v16.4s, v6.4s,v24.s[0]
	fmla v14.4s, v6.4s,v26.s[0]
	sub x16,x10,#32
    prfm  pldl1keep, [x16,x15]   //row 4_0

    fmla v16.4s, v7.4s,v24.s[1]
	fmla v14.4s, v7.4s,v26.s[1]
    add x16,x16,x9, LSL #1
    prfm  pldl1keep, [x16,x15]  // row 4_2

    fmla v16.4s, v27.4s,v24.s[2]
    fmla v14.4s, v27.4s,v26.s[2]

	sub x16,x12,#32              //row 4_1
    prfm  pldl1keep, [x16,x15]

 //add bias
    fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_102f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_102f
	fmin v14.4s,v14.4s,v29.4s

save_done_102f:
#endif
    add x16,x16,x9, LSL #1
    prfm  pldl1keep, [x16,x15]  // row 4_3

    str q14,[x8],#16
    str q16,[x6],#16

	b first_row_2_loop_4

first_row_2_loop_2:

    sub x14,x2,#5
    cmp x14,x13
	blt first_row_2_last_4_3_2

    movi d1,#0
    movi d7,#0

    ldr q0,[x10],#16
    ldr s1,[x10]
    ldr q6,[x12],#16
    ldr s7,[x12]

    uzp1  v2.4s,v0.4s,v1.4s
    uzp2  v3.4s,v0.4s,v1.4s
    ext   v4.16b,v2.16b,v31.16b,#4

    fmul v14.4s, v2.4s,v25.s[0]
	uzp1  v8.4s,v6.4s,v7.4s
    fmla v14.4s, v3.4s,v25.s[1]
    uzp2  v9.4s,v6.4s,v7.4s
    fmla v14.4s, v4.4s,v25.s[2]
	ext   v10.16b,v8.16b,v31.16b,#4

    fmul v16.4s, v8.4s,v24.s[0]
	fmla v14.4s, v8.4s,v26.s[0]
    fmla v16.4s, v9.4s,v24.s[1]
	fmla v14.4s, v9.4s,v26.s[1]
    fmla v16.4s, v10.4s,v24.s[2]
    fmla v14.4s, v10.4s,v26.s[2]

//add bias
    fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_103f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_103f
    fmin v14.4s,v14.4s,v29.4s
save_done_103f:

#endif
    str  d14,[x8],#8
    str  d16,[x6],#8

	add x13,x13,#4
	b first_row_2_loop_2

first_row_2_last_4_3_2:
    sub x14,x2,x13
    cmp x14,#4
	beq first_row_2_last_4
	cmp x14,#3
	beq first_row_2_last_3
	cmp x14,#2
	beq first_row_2_last_2
	cmp x14,#2
	blt first_row_2_end

first_row_2_last_4:

   ldr q0,[x10]
   ldr q6,[x12]

   uzp1  v2.4s,v0.4s,v31.4s
   uzp2  v3.4s,v0.4s,v31.4s
   ext   v4.16b,v2.16b,v31.16b,#4

   fmul v14.4s, v2.4s,v25.s[0]
   uzp1  v8.4s,v6.4s,v31.4s
   fmla v14.4s, v3.4s,v25.s[1]
   uzp2  v9.4s,v6.4s,v31.4s
   fmla v14.4s, v4.4s,v25.s[2]
   ext   v10.16b,v8.16b,v31.16b,#4

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
	blt save_done_104f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_104f
	fmin v14.4s,v14.4s,v29.4s
save_done_104f:
#endif
   str  d14,[x8]
   str  d15,[x6]

   b first_row_2_end

first_row_2_last_3:
   ldp s0,s1,[x10]
   ldr s2,[x10,#8]
   ldp s6,s7,[x12]
   ldr s8,[x12,#8]

   fmul s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1]
   fmla s14, s2,v25.s[2]

   fmul s15, s6,v24.s[0]
   fmla s14, s6,v26.s[0]
   fmla s15, s7,v24.s[1]
   fmla s14, s7,v26.s[1]
   fmla s15, s8,v24.s[2]
   fmla s14, s8,v26.s[2]

//add bias
    fadd s14,s14,s21
#ifdef CONV_RELU_FUSE
	cmp w18,0
    blt save_done_105f
    fmax s14,s14,s31
	beq save_done_105f
	fmin s14,s14,s29
save_done_105f:
#endif
   str s14,[x8]
   str s15,[x6]

   b first_row_2_end
first_row_2_last_2:
   ldp s0,s1,[x10]
   ldp s6,s7,[x12]

   fmul s14, s0,v25.s[0]
   fmla s14, s1,v25.s[1]
   fmul s15, s6,v24.s[0]
   fmla s14, s6,v26.s[0]
   fmla s15, s7,v24.s[1]
   fmla s14, s7,v26.s[1]
//add bias
    fadd s14,s14,s21
#ifdef CONV_RELU_FUSE
	cmp w18,0
    blt save_done_106f
    fmax s14,s14,s31
	beq save_done_106f
	fmin s14,s14,s29
save_done_106f:
#endif
   str s14,[x8]
   str s15,[x6]

first_row_2_end:
   add x4,x4,x7
   mov x8,x4
   add x6,x4,x7
   add x11,x11,#2
   add x0,x0,x9,LSL #1
   mov x10,x0
   add x12,x0,x9
   mov x13,#0

   LSL x15,x15,#1            // preload next 4 rows

mid_row_4_loop:
   subs x14,x1,x11
   cmp x14,#4
   ble mid_row_2_loop

mid_row_4_first_4:

	ldr q14,[x8]               //get the tmp value

    add x12,x10,x9
	add x14,x12,x9

	ld2 {v0.4s,v1.4s},[x10]     // line 0
	ld2 {v2.4s,v3.4s},[x12]     // line 1
	ld2 {v6.4s,v7.4s},[x14]     // line 2

	ext v5.16b,v31.16b,v1.16b,  #12
	ext v27.16b,v31.16b,v3.16b,  #12

	prfm  pldl1keep, [x10,x15]     // row 4_0

	fmla v14.4s, v5.4s,v25.s[0]
	fmul v15.4s, v27.4s,v24.s[0]
	prfm  pldl1keep, [x12,x15]     // row 4_1

    fmla v14.4s, v0.4s,v25.s[1]
	fmla v15.4s, v2.4s, v24.s[1]
	prfm  pldl1keep, [x14,x15]    // row 4_2

    fmla v14.4s, v1.4s,v25.s[2]
	fmla v15.4s, v3.4s, v24.s[2]

	ext v13.16b,v31.16b,v7.16b,  #12
	add x12,x14,x9               //input row 3

    fmla v14.4s, v27.4s,v26.s[0]
	fmla v15.4s, v13.4s, v25.s[0]
	ld2 {v8.4s,v9.4s},[x12]

    fmla v14.4s, v2.4s, v26.s[1]
	fmla v15.4s, v6.4s, v25.s[1]
    ext v28.16b,v31.16b,v9.16b,  #12

    fmla v14.4s, v3.4s, v26.s[2]
    fmla v15.4s, v7.4s, v25.s[2]

	prfm  pldl1keep, [x12,x15]  // row 4_3

    fmla v15.4s, v28.4s, v26.s[0]
	fmul v16.4s, v28.4s, v24.s[0]
    fadd v14.4s,v14.4s,v21.4s   // add bias

    fmla v15.4s, v8.4s,  v26.s[1]
	fmla v16.4s, v8.4s,  v24.s[1]
    add x13,x13,#7

    fmla v15.4s, v9.4s,  v26.s[2]
    fmla v16.4s, v9.4s,  v24.s[2]
    fadd v15.4s,v15.4s,v21.4s   // add bias

#ifdef CONV_RELU_FUSE
    cmp w18,0
	blt save_done_006f

	fmax v14.4s,v14.4s,v31.4s
	fmax v15.4s,v15.4s,v31.4s

	beq save_done_006f
	fmin v14.4s,v14.4s,v29.4s
	fmin v15.4s,v15.4s,v29.4s

save_done_006f:
#endif

    str q14,[x8]      //output line 0
	add x6,x8,x7
    str q15,[x6]      //output line 1
	add x6,x6,x7
	str q16,[x6]      //ouput  line 2

	add x8,x8,#16     //output offset 4
    add x10,x10,#28

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
	blt save_done_107f

    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
	fmax v16.4s,v16.4s,v31.4s
    fmax v17.4s,v17.4s,v31.4s

	beq save_done_107f

    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s
	fmin v16.4s,v16.4s,v29.4s
    fmin v17.4s,v17.4s,v29.4s

save_done_107f:


#endif
    stp q14,q15,[x8]
	add x6,x8,x7
    stp q16,q17,[x6]
	add x6,x6,x7
	add x8,x8,#32
	add x13,x13,#16
	stp q18,q19,[x6]

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
	blt save_done_108f

	fmax v14.4s,v14.4s,v31.4s
	fmax v15.4s,v15.4s,v31.4s

	beq save_done_108f
	fmin v14.4s,v14.4s,v29.4s
	fmin v15.4s,v15.4s,v29.4s
save_done_108f:

#endif
    str q14,[x8]
	add x6,x8,x7
    str q15,[x6]
	add x6,x6,x7
	add x8,x8,#16
	add x13,x13,#8
    str q16,[x6]

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
	blt save_done_109f

	fmax v14.4s,v14.4s,v31.4s
	fmax v15.4s,v15.4s,v31.4s

	beq save_done_109f
	fmin v14.4s,v14.4s,v29.4s
	fmin v15.4s,v15.4s,v29.4s

save_done_109f:
#endif
    str d14,[x8]
	add x6,x8,x7
    str d15,[x6]
	add x6,x6,x7
	add x8,x8,#8
	add x13,x13,#4
    str d16,[x6]

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
   blt save_done_110f

   fmax v14.4s,v14.4s,v31.4s
   fmax v15.4s,v15.4s,v31.4s

   beq save_done_110f
   fmin v14.4s,v14.4s,v29.4s
   fmin v15.4s,v15.4s,v29.4s

save_done_110f:

#endif
   str d14,[x8]
   add x6,x8,x7
   str d15,[x6]
   add x6,x6,x7
   str d16,[x6]

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
   blt save_done_111f
   fmax s14,s14,s31
   fmax s15,s15,s31
   beq save_done_111f
   fmin s14,s14,s29
   fmin s15,s15,s29
save_done_111f:
#endif
   str s14,[x8]
   add x6,x8,x7
   str s15,[x6]
   add x6,x6,x7
   str s16,[x6]

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
	blt save_done_112f

    fmax s14,s14,s31
	fmax s15,s15,s31

	beq save_done_112f

	fmin s14,s14,s29
	fmin s15,s15,s29

save_done_112f:

#endif
   str s14,[x8]
   add x6,x8,x7
   str s15,[x6]
   add x6,x6,x7
   str s16,[x6]
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
   add x6,x8,x7

   subs x14,x1,x11
   cmp x14,#2
   beq last_row_2_loop
   cmp x14,#1
   beq last_row_loop

mid_row_2_first_4:

    ldr q14,[x8]
	ld2 {v0.4s,v1.4s},[x10]   //first time shift 7
	ld2 {v2.4s,v3.4s},[x12]


	ext v5.16b,v31.16b,v1.16b,  #12
	ext v27.16b,v31.16b,v3.16b,  #12

    fmla v14.4s, v5.4s,v25.s[0]
    add x13,x13,#7
    fmla v14.4s, v0.4s,v25.s[1]
	add x10,x10,#28
    fmla v14.4s, v1.4s,v25.s[2]
	add x12,x12,#28

    fmla v14.4s, v27.4s,v26.s[0]
	fmul v15.4s, v27.4s,v24.s[0]
    fmla v14.4s, v2.4s, v26.s[1]
	fmla v15.4s, v2.4s, v24.s[1]
    fmla v14.4s, v3.4s, v26.s[2]
    fmla v15.4s, v3.4s, v24.s[2]

//add bias
    fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_113f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_113f
	fmin v14.4s,v14.4s,v29.4s

save_done_113f:

#endif
    str q14,[x8],#16
    str q15,[x6],#16

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
	blt save_done_114f
    fmax v14.4s,v14.4s,v31.4s
    fmax v15.4s,v15.4s,v31.4s
	beq save_done_114f
    fmin v14.4s,v14.4s,v29.4s
    fmin v15.4s,v15.4s,v29.4s

save_done_114f:

#endif
    stp q14,q15,[x8],#32
    stp q16,q17,[x6],#32

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
	blt save_done_115f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_115f
	fmin v14.4s,v14.4s,v29.4s

save_done_115f:
#endif
    str q14,[x8],#16
    str q16,[x6],#16

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
	blt save_done_116f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_116f
	fmin v14.4s,v14.4s,v29.4s

save_done_116f:

#endif
    str d14,[x8],#8
    str d16,[x6],#8

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
	blt save_done_117f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_117f
	fmin v14.4s,v14.4s,v29.4s

save_done_117f:

#endif
   str d14,[x8]
   str d15,[x6]

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
	blt save_done_120f
    fmax s14,s14,s31
	beq save_done_120f
	fmin s14,s14,s29
save_done_120f:
#endif
   str s14,[x8]
   str s15,[x6]

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
	blt save_done_121f
    fmax s14,s14,s31
	beq save_done_121f
    fmin s14,s14,s29

save_done_121f:
#endif
   str s14,[x8]
   str s15,[x6]
mid_row_2_end:
   add x4,x4,x7
   mov x8,x4
   add x6,x4,x7
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

last_row_2_first_4:
    ldr q14,[x8]

	ld2 {v0.4s,v1.4s},[x10]   //first time shift 7
	ld2 {v2.4s,v3.4s},[x12]

	ext v5.16b,v31.16b,v1.16b,   #12
    fmla v14.4s, v5.4s,v25.s[0]
	prfm  pldl1keep, [x10,x15]
    fmla v14.4s, v0.4s,v25.s[1]
	ext v27.16b,v31.16b,v3.16b,  #12
    fmla v14.4s, v1.4s,v25.s[2]
    add x10,x10,#28
    prfm  pldl1keep, [x12,x15]
    fmla v14.4s, v27.4s,v26.s[0]
	add x12,x12,#28
    fmla v14.4s, v2.4s, v26.s[1]
    add x13,x13,#7
    fmla v14.4s, v3.4s, v26.s[2]

//add bias
    fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_122f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_122f
	fmin v14.4s,v14.4s,v29.4s

save_done_122f:
#endif
    str q14,[x8],#16

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
	blt save_done_123f
    fmax v14.4s,v14.4s,v31.4s
	fmax v15.4s,v15.4s,v31.4s
	beq save_done_123f
    fmin v14.4s,v14.4s,v29.4s
	fmin v15.4s,v15.4s,v29.4s
save_done_123f:

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
	blt save_done_125f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_125f
	fmin v14.4s,v14.4s,v29.4s

save_done_125f:

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
	blt save_done_126f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_126f
	fmin v14.4s,v14.4s,v29.4s

save_done_126f:
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
	blt save_done_127f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_127f
	fmin v14.4s,v14.4s,v29.4s

save_done_127f:
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
	blt save_done_128f
    fmax s14,s14,s31
	beq save_done_128f
	fmin s14,s14,s29

save_done_128f:
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
	blt save_done_129f
    fmax s14,s14,s31
	beq save_done_129f
	fmin s14,s14,s29

save_done_129f:
#endif
   str s14,[x8]

   b last_row_end

last_row_loop:
   mov x10,x0
   add x12,x0,x9
   mov  x15,#8
   mul  x15,x2,x15
last_row_first_4:
   //line 0
	ld2 {v0.4s,v1.4s},[x10]   //first time shift 7
	ext v5.16b,v31.16b,v1.16b,  #12

    ldr q14,[x8]
    prfm  pldl1keep, [x10,x15]
    fmla v14.4s, v5.4s,v25.s[0]
	add x13,x13,#7
    fmla v14.4s, v0.4s,v25.s[1]
    add x10,x10,#28
    fmla v14.4s, v1.4s,v25.s[2]
	add x12,x12,#28
//add bias
    fadd v14.4s,v14.4s,v21.4s
#ifdef CONV_RELU_FUSE
	cmp w18,0
	blt save_done_130f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_130f
	fmin v14.4s,v14.4s,v29.4s

save_done_130f:

#endif
    str q14,[x8],#16

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
	blt save_done_131f
    fmax v14.4s,v14.4s,v31.4s
	fmax v15.4s,v15.4s,v31.4s
	beq save_done_131f
    fmin v14.4s,v14.4s,v29.4s
	fmin v15.4s,v15.4s,v29.4s

save_done_131f:
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
	blt save_done_132f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_132f
	fmin v14.4s,v14.4s,v29.4s

save_done_132f:

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
	blt save_done_133f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_133f
    fmin v14.4s,v14.4s,v29.4s

save_done_133f:
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
	blt save_done_134f
    fmax v14.4s,v14.4s,v31.4s
	beq save_done_134f
	fmin v14.4s,v14.4s,v29.4s

save_done_134f:
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
	blt save_done_135f
    fmax s14,s14,s31
	beq save_done_135f
	fmin s14,s14,s29

save_done_135f:
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
	blt save_done_136f
    fmax s14,s14,s31
	beq save_done_136f
	fmin s14,s14,s29

save_done_136f:
#endif
   str s14,[x8]
last_row_end:
     ldp d8,d9,[sp]
     ldp d10,d11,[sp,0x10]
     ldp d12,d13,[sp,0x20]
     ldp d14,d15,[sp,0x30]
     add sp,sp,#0x40

    ret



