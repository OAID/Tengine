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

// register definition
// x0        output start address
// x1        input start address
// x2        kernel start address
// x3        cin
// x4        direct save/stride save

// x9 ~ x10  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_4x4_A72 STT_FUNC
    .global wino_sgemm_4x4_A72
    .hidden wino_sgemm_4x4_A72
    
wino_sgemm_4x4_A72:
	cmp	x3, 0x4

none_biases:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0


start:
	and	x10,x3, 0x3
	b.lt	loop4_end
	lsr	x9, x3, 0x2

loop4:  
	ldr	q0, [x1]			// q0=i[3-0] q1=i[3-0]
	ldr	q4, [x2]			// q4=k[3-0] q5=k[7-4] 

	ldr	q1, [x1, 0x10]			// q0=i[3-0] q1=i[3-0]
	ldr	q5, [x2, 0x10]			// q4=k[3-0] q5=k[7-4] 

	ldp	q2, q3, [x1, 0x20]		// q2=i[3-0] q3=i[3-0]
	ldp	q6, q7, [x2, 0x20]		// q6=k[b-8] q7=k[f-c]

    subs	x9, x9, 0x1

	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]

	fmla	v16.4s, v1.4s,  v5.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v5.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v1.4s,  v5.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v5.s[3]		// i[3-0]k[3]

	fmla	v16.4s, v2.4s, v6.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v2.4s, v6.s[1]		// i[3-0]k[1]
	prfm	pldl1keep, [x1, 0x140]
    fmla	v18.4s, v2.4s, v6.s[2]		// i[3-0]k[2]
	prfm	pldl1keep, [x2, 0x140]
	fmla	v19.4s, v2.4s, v6.s[3]		// i[3-0]k[3]

	fmla	v16.4s, v3.4s, v7.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v3.4s, v7.s[1]		// i[3-0]k[1]
    add	x1, x1, 0x40
	fmla	v18.4s, v3.4s, v7.s[2]		// i[3-0]k[2]
    add	x2, x2, 0x40
	fmla	v19.4s, v3.4s, v7.s[3]		// i[3-0]k[3]

	b.ne	loop4

loop4_end:
	cbz	x10, save_result

loop1:
	ldr     q0, [x1],0x10                    // q0=i[3-0]
    ldr     q4, [x2], 0x10                  // q4=k[3-0]
	subs	x10 ,x10 ,0x1
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[0]k[3-0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[1]k[3-0]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[2]k[3-0]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3]k[3-0]
        
    b.ne    loop1
	
save_result:
    cmp     w4,0
    beq     direct_save

    stride_save:
    mov x10,#0x240
   
    str q16,[x0]
    str q17,[x0,#0x240]   //each line 36*4 data
    str q18,[x0,#0x480]
    str q19,[x0,#0x6c0]

    b end_func
    
    direct_save:
    stp  q16,q17, [x0]
    stp	 q18,q19, [x0, 0x20]


end_func:


	ret
        .end

