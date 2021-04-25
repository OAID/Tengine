/*
 * Licensed to the Apache Software Foundation (ASF) under One
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
 * Author: haitao@openailab.com
 */

//
// find max data
//
//
// input:
//         r0   arg0   data start address
//         r1   arg1   input data size
//         r2   arg2   max data address
//
// output: no
//
//
        .section .text,"ax"
        .align 5

        .type get_max_arm32 STT_FUNC
        .global get_max_arm32
        .hidden get_max_arm32
get_max_arm32:
	vmov.i64	q0, #0x0
	pld		[r0]
	vmov.i64	q1, #0x0
	pld		[r0, #0x40]
	vmov.i64	q2, #0x0
	pld		[r0, #0x80]
	vmov.i64	q3, #0x0

	cmp		r1, #16
	blt		last16
	pld		[r0, #0xc0]

	lsr		r3, r1, #4

loop16:
	vldm		r0!, {d16-d23}
	vabs.f32	q8, q8
	vabs.f32	q9, q9
	vabs.f32	q10,q10
	vabs.f32	q11,q11
	vmax.f32	q0, q0, q8
	vmax.f32	q1, q1, q9
	vmax.f32	q2, q2, q10
	vmax.f32	q3, q3, q11
	subs		r3, r3, #1
	bne		loop16

	vmax.f32	q0, q0, q1
	vmax.f32	q2, q2, q3
	vmax.f32	q0, q0, q2
	vmax.f32	d0, d0, d1
	vmov		s2, s1
	vmax.f32	d0, d0, d1

	and		r1, r1, #0xf	

last16:
	cmp		r1, #0
	beq		save_result

loop1:
	vldm		r0!, {s2}
	vabs.f32	s2, s2
	vmax.f32	d0, d0, d1
	subs		r1, r1, #1
	bne		loop1


save_result:
	vstr		s0, [r2]
	bx		lr

        .end
