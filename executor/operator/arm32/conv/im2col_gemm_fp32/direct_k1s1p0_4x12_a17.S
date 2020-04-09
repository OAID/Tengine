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
// x0        bias start address
// x1        input start address
// x2        kernel start address
// x3        output start address
// x4        in_hw
// x5        c_in
//           activation

// double pld for r1,r6,r7,r8
// v0       input
// v1 v2 v3 kernel
// v4 - v15 output


	.section .text,"ax"
	.align 5

	.type direct_k1s1p0_4x12_a17 STT_FUNC
	.global direct_k1s1p0_4x12_a17
	.hidden direct_k1s1p0_4x12_a17

direct_k1s1p0_4x12_a17:

    // context save & load parameter
	push		{r4 - r8, lr}               // [6 num]
	vpush		{d8 - d15}                  //  [8dx2=16 num]
	ldr		r4,[sp, #0x58]	// r4 = in_hw    [88=22num*4]
    ldr		r5,[sp, #0x5c]	// r5 = c_in     [92 = 88+4]
	                        //  0x60 activation   [96 =92+4]

	teq	r0, #0x0
    beq none_biases

        vld4.32		{d8[], d10[], d12[], d14[]}, [r0]!
        vmov		d9,  d8
        vmov		d11, d10
        vld4.32		{d16[],d18[], d20[], d22[]}, [r0]!
        vmov		d13, d12
        vmov		d15, d14
        vld4.32		{d24[],d26[], d28[], d30[]}, [r0]!
        vmov		d17, d16
        vmov		d19, d18
        vmov		d21, d20
        vmov		d23, d22
        vmov		d25, d24
        vmov		d27, d26
        vmov		d29, d28
        vmov		d31, d30
	    b convolution_start

none_biases:
	vmov.i64	q4,  #0x0
	vmov.i64	q5,  #0x0
	vmov.i64	q6,  #0x0
	vmov.i64	q7,  #0x0
	vmov.i64	q8,  #0x0
	vmov.i64	q9,  #0x0
	vmov.i64	q10, #0x0
	vmov.i64	q11, #0x0
	vmov.i64	q12, #0x0
	vmov.i64	q13, #0x0
	vmov.i64	q14, #0x0
	vmov.i64	q15, #0x0

convolution_start:
    lsl r4,r4, #0x2
    cmp r5,#0x4                          // if c_in <4, go to loop4_end
    blt loop4_end
   

    add r6, r1, r4
	lsr	r0, r5, #0x2                          // X0 = c_in / 4
    add r7, r6, r4
    add r8, r7, r4

loop4:  
    vldm		r1,{d0-d1}		        //inp0
	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		//ker1
	subs		r0, r0, #1
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vldm		r2!,{d4-d5}		//ker2
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1] 
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		//ker3
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]  
	vldm		r6,{d6-d7}		         //inp1
	vmla.f32	q12,q0, d2[0]
	vmla.f32	q13,q0, d2[1]
    vldm		r2!,{d4-d5}		//ker1-1
	vmla.f32	q14,q0, d3[0]
	vmla.f32	q15,q0, d3[1]  




	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		//ker2-2
	vmla.f32	q4, q3, d4[0]   
	vmla.f32	q5, q3, d4[1]

	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]   
    vldm		r2!,{d4-d5}		//ker3-3
	vmla.f32	q8, q3, d2[0]
	vmla.f32	q9, q3, d2[1]
	vmla.f32	q10,q3, d3[0]
	vmla.f32	q11,q3, d3[1]  
	vldm		r7,{d0-d1}   	          // inp2
	vmla.f32	q12,q3, d4[0]
	vmla.f32	q13,q3, d4[1]
	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		// ker1-1-1
	vmla.f32	q14,q3, d5[0]
	vmla.f32	q15,q3, d5[1]  





	vldm		r2!,{d4-d5}		// ker2-2-2
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]   
	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		// ker3-3-3
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]

	vldm		r2!,{d4-d5}		// ker1-1-1-1
	vmla.f32	q12,q0, d2[0]
	vmla.f32	q13,q0, d2[1]

	vldm		r8,{d6-d7}					// inp3
	vmla.f32	q14,q0, d3[0]
	vmla.f32	q15,q0, d3[1]

	pld			[r2,#0x100]
    vldm		r2!,{d2-d3}		// ker2-2-2-2
	vmla.f32	q4, q3, d4[0]
	pld		[r1,#0x40]
	vmla.f32	q5, q3, d4[1]
	add  r1,r1,r4, LSL #0x2
	pld		[r1,#0x40]
	vmla.f32	q6, q3, d5[0]
	vmla.f32	q7, q3, d5[1]
	vldm		r2!,{d4-d5}		// ker3-3-3-3
	vmla.f32	q8, q3, d2[0]
	pld		[r6,#0x40]
	vmla.f32	q9, q3, d2[1]
	add  r6,r6,r4, LSL #0x2
	pld		[r6,#0x40]
	vmla.f32	q10,q3, d3[0]
	pld		[r7,#0x40]
	vmla.f32	q11,q3, d3[1]
	
	add  r7,r7,r4, LSL #0x2
 	pld		[r7,#0x40]
	vmla.f32	q12,q3, d4[0]
	pld		[r8,#0x40]
	vmla.f32	q13,q3, d4[1]
	add  r8,r8,r4, LSL #0x2
	pld		[r8,#0x40]
	vmla.f32	q14,q3, d5[0]
	vmla.f32	q15,q3, d5[1]

	bne		loop4
   
 

loop4_end:
	ldr		r0, [sp, #0x60]   // activation
    ands r6,r5,#0x3 
    beq activation

loop1:
    vldm       r1,{d0-d1}		                                               
    vldm        r2!,{d2-d7}  
	subs r6,r6,#0x1                                     
	vmla.f32	q4, q0, d2[0]
	vmla.f32	q5, q0, d2[1]
	vmla.f32	q6, q0, d3[0]
	vmla.f32	q7, q0, d3[1]
	vmla.f32	q8, q0, d4[0]
	vmla.f32	q9, q0, d4[1]
	add 		r1,r1,r4
	vmla.f32	q10,q0, d5[0]
	vmla.f32	q11,q0, d5[1]
	vmla.f32	q12,q0, d6[0]
	vmla.f32	q13,q0, d6[1]
	vmla.f32	q14,q0, d7[0]
	vmla.f32	q15,q0, d7[1]
    bne loop1

activation:
	cmp         r0, #0x0
	
        blt         save_result
	
        vmov.i64	q0, #0x0
	vmax.f32	q4, q4, q0
	vmax.f32	q5, q5, q0
	vmax.f32	q6, q6, q0
	vmax.f32	q7, q7, q0
	vmax.f32	q8, q8, q0
	vmax.f32	q9, q9, q0
	vmax.f32	q10,q10,q0
	vmax.f32	q11,q11,q0
	vmax.f32	q12,q12,q0
	vmax.f32	q13,q13,q0
	vmax.f32	q14,q14,q0
	vmax.f32	q15,q15,q0

        beq         save_result
	
        vdup.32	        q0, r0
        vcvt.f32.s32    q0, q0
	vmin.f32	q4, q4, q0
	vmin.f32	q5, q5, q0
	vmin.f32	q6, q6, q0
	vmin.f32	q7, q7, q0
	vmin.f32	q8, q8, q0
	vmin.f32	q9, q9, q0
	vmin.f32	q10,q10,q0
	vmin.f32	q11,q11,q0
	vmin.f32	q12,q12,q0
	vmin.f32	q13,q13,q0
	vmin.f32	q14,q14,q0
	vmin.f32	q15,q15,q0

save_result:
                                        // r3
    add     r0, r3, r4              // r0 = + hw
    add     r5, r3, r4, LSL #0x1    // r5 = + 2*hw
    add     r6, r0, r4, LSL #0x1    // r6 = + 3*hw

	vstm		r3, {d8,d9}
    add     r3,r3,r4, LSL #0x2
    vstm		r0, {d10,d11}
    add     r0,r0,r4, LSL #0x2
    vstm		r5, {d12,d13}
    add     r5,r5,r4, LSL #0x2
    vstm		r6, {d14,d15}
    add     r6,r6,r4, LSL #0x2

    vstm		r3, {d16,d17}
    add     r3,r3,r4, LSL #0x2
    vstm		r0, {d18,d19}
    add     r0,r0,r4, LSL #0x2
    vstm		r5, {d20,d21}
    add     r5,r5,r4, LSL #0x2
    vstm		r6, {d22,d23}
    add     r6,r6,r4, LSL #0x2

	vstm		r3, {d24,d25}
	vstm		r0, {d26,d27}
	vstm		r5, {d28,d29}
	vstm		r6, {d30,d31}

	// restore content
	vpop		{d8 - d15}    
	pop		{r4 - r8, pc}
	
       .end
