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

// x9 ~ x10  temp loop counter

    .section .text,"ax"
    .align 5

    .type wino_sgemm_4x16_A72 STT_FUNC
    .global wino_sgemm_4x16_A72
    .hidden wino_sgemm_4x16_A72
    
wino_sgemm_4x16_A72:
	// bring some code ahead to reduce dependency
	prfm	pldl1keep, [x1]
	cmp	x3, 0x4

none_biases:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0
	movi	d20, 0x0
	movi	d21, 0x0
	movi	d22, 0x0
	movi	d23, 0x0
	movi	d24, 0x0
	movi	d25, 0x0
	movi	d26, 0x0
	movi	d27, 0x0
	movi	d28, 0x0
	movi	d29, 0x0
	movi	d30, 0x0
	movi	d31, 0x0

start:
	ldr	q0, [x1]			// q0=i[3-0]
	ldp	q4, q5, [x2]	    // q4=k[3-0] q5=k[7-4] 
	and	x10,x3, 0x3
	b.lt	loop4_end
	lsr	x9, x3, 0x2

loop4:  
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	ldp	q6, q7, [x2, 0x20]		// q6=k[b-8] q7=k[f-c]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	subs	x9, x9, 0x1
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q1, [x1, 0x10]			// q1=i[3-0]
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x2, 0x40]		// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	prfm	pldl1keep, [x1, 0x80]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	prfm	pldl1keep, [x2, 0x140]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x2, 0x60]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v1.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v1.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q0, [x1, 0x20]			// q1=i[3-0]
	fmla	v20.4s, v1.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v1.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v1.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v1.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x2, 0x80]		// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v1.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v1.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v1.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v1.4s,  v6.s[3]		// i[3-0]k[b]
	prfm	pldl1keep, [x2, 0x180]
	fmla	v28.4s, v1.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v1.4s,  v7.s[1]		// i[3-0]k[d]
	prfm	pldl1keep, [x2, 0x1c0]
	fmla	v30.4s, v1.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v1.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x2, 0xa0]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q1, [x1, 0x30]			// q1=i[3-0]
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
	add	x1, x1, 0x40
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x2, 0xc0]		// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	prfm	pldl1keep, [x2, 0x200]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]

	ldp	q6, q7, [x2, 0xe0]		// q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v1.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v1.4s,  v4.s[1]		// i[3-0]k[1]
	add	x2, x2, 0x100
	fmla	v18.4s, v1.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v1.4s,  v4.s[3]		// i[3-0]k[3]
	ldr	q0, [x1]			// q0=i[3-0]
	fmla	v20.4s, v1.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v1.4s,  v5.s[1]		// i[3-0]k[5]
	fmla	v22.4s, v1.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v1.4s,  v5.s[3]		// i[3-0]k[7]
	ldp	q4, q5, [x2]			// q4=k[3-0] q5=k[7-4] 
	fmla	v24.4s, v1.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v1.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v1.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v1.4s,  v6.s[3]		// i[3-0]k[b]
	fmla	v28.4s, v1.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v1.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v1.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v1.4s,  v7.s[3]		// i[3-0]k[f]
	b.ne	loop4


loop4_end:
	cbz	x10, save_result

loop1:
        ldp     q6, q7, [x2, 0x20]              // q6=k[b-8] q7=k[f-c]
	fmla	v16.4s, v0.4s,  v4.s[0]		// i[3-0]k[0]
	fmla	v17.4s, v0.4s,  v4.s[1]		// i[3-0]k[1]
        add     x2, x2, 0x40
	fmla	v18.4s, v0.4s,  v4.s[2]		// i[3-0]k[2]
	fmla	v19.4s, v0.4s,  v4.s[3]		// i[3-0]k[3]
	add	x1, x1, 0x10
	fmla	v20.4s, v0.4s,  v5.s[0]		// i[3-0]k[4]
	fmla	v21.4s, v0.4s,  v5.s[1]		// i[3-0]k[5]
        subs    x10, x10 ,0x1
	fmla	v22.4s, v0.4s,  v5.s[2]		// i[3-0]k[6]
	fmla	v23.4s, v0.4s,  v5.s[3]		// i[3-0]k[7]
	ldp     q4, q5, [x2]                    // q4=k[3-0] q5=k[7-4]
	fmla	v24.4s, v0.4s,  v6.s[0]		// i[3-0]k[8]
	fmla	v25.4s, v0.4s,  v6.s[1]		// i[3-0]k[9]
	fmla	v26.4s, v0.4s,  v6.s[2]		// i[3-0]k[a]
	fmla	v27.4s, v0.4s,  v6.s[3]		// i[3-0]k[b]
	fmla	v28.4s, v0.4s,  v7.s[0]		// i[3-0]k[c]
	fmla	v29.4s, v0.4s,  v7.s[1]		// i[3-0]k[d]
	fmla	v30.4s, v0.4s,  v7.s[2]		// i[3-0]k[e]
	fmla	v31.4s, v0.4s,  v7.s[3]		// i[3-0]k[f]
        ldr     q0, [x1]                        // q0=i[3-0]

    b.ne    loop1
	
	
save_result:
    cmp     w4,0
    beq     direct_save

    stride_save:
    mov x10,#0x240
    lsl x11,x10,#2

    add x6,x0,x11
    add x7,x0,x11,lsl #1
    add x8,x7,x11
    str q16,[x0]
    str q17,[x0,#0x240]   //each line 36*4 data
    str q18,[x0,#0x480]
    str q19,[x0,#0x6c0]
    str q20,[x6]
    str q21,[x6,#0x240]   //each line 36*4 data
    str q22,[x6,#0x480]
    str q23,[x6,#0x6c0]
    str q24,[x7]
    str q25,[x7,#0x240]   //each line 36*4 data
    str q26,[x7,#0x480]
    str q27,[x7,#0x6c0]
    str q28,[x8]
    str q29,[x8,#0x240]   //each line 36*4 data
    str q30,[x8,#0x480]
    str q31,[x8,#0x6c0]

    b end_func
    
    direct_save:
    stp  q16,q17, [x0]
    stp	 q18,q19, [x0, 0x20]
    stp	 q20,q21, [x0, 0x40]
    stp	 q22,q23, [x0, 0x60]
    stp	 q24,q25, [x0, 0x80]
    stp	 q26,q27, [x0, 0xa0]
    stp	 q28,q29, [x0, 0xc0]
    stp	 q30,q31, [x0, 0xe0]


end_func:

	ret
        .end

