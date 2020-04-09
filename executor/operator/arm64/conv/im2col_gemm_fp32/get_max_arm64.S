/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor License agreements.  See the NOTICE file
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
//         x0   arg0   data start address
//         x1   arg1   input data size
//         x2   arg2   max data address
//  
// output: no
//
//

        .section .text,"ax"
        .align 5

        .type get_max_arm64 STT_FUNC
        .global get_max_arm64
        .hidden get_max_arm64
get_max_arm64:
	// main loop start
	movi	d0, 0x0	

	cmp	x1, 16
	blt	last16

	movi	d1, 0x0	
	movi	d2, 0x0	
	movi	d3, 0x0	
	lsr	x3, x1, 4

loop16:
	ldp	q4, q5, [x0]
	ldp	q6, q7, [x0, 0x20]

	fabs	v4.4s, v4.4s
	fabs	v5.4s, v5.4s
	fabs	v6.4s, v6.4s
	fabs	v7.4s, v7.4s
	fmax	v0.4s, v0.4s, v4.4s
	fmax	v1.4s, v1.4s, v5.4s
	fmax	v2.4s, v2.4s, v6.4s
	fmax	v3.4s, v3.4s, v7.4s
	add	x0, x0, 0x40
	subs	x3, x3, 1
	bne	loop16
	
	fmax	v0.4s, v1.4s, v0.4s
	fmax	v2.4s, v2.4s, v3.4s
	fmax	v0.4s, v0.4s, v2.4s
	fmaxv	s0, v0.4s
	and	x1, x1, 0xf

last16:
	cbz	x1, save_result

loop1:
	ldr	s4, [x0], 0x4
	fabs	s4, s4
	fmax	s0, s0, s4
	subs	x1, x1, 1
	bne	loop1

save_result:
	str	s0, [x2]

	ret

        .end
