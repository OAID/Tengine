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
 * Author: xiaowei@openailab.com
 */
//
// im2col for kernel 1x1 s1p0d1
//
// input:
//         x0 arg0  input address 
//         x1 arg1  input_xy
//         x2 arg2  col address
//         x3 arg3  col_cnt must be multiply of 4
//         x4 arg4  input channel
//
// register definition
//    x0 input address 
//    x1 input_xy x 4
//    x2 col address
//    x3 col_cnt
//    x4 input channel
//    x6 input start pointer
//    x7 input pointer
//    x9 channel cnt
//    x11

        .section .text,"ax"
        .align 5

        .type   im2col_fp32_1x1 STT_FUNC
        .global im2col_fp32_1x1
        .hidden im2col_fp32_1x1
im2col_fp32_1x1:
	cmp	x3, 4
	b.lt	col_end
	lsr	x3, x3, 2	// x3 = col_cnt
	lsl	x1, x1, 2	// x1 = input_xy size
	mov	x6, x0
	lsl	x12,x1, 1	// x12 = input_xy size * 2
	and	x10,x4, 1

	// col loop
col_loop:
	mov	x7, x6		// x7 = input
	lsr	x9, x4, 1	// x9 = channel cnt
	cbz	x9, channel_last
	add	x11,x7, x1
	// kernel size loop
channel_loop2:
	ldr	q0, [x7]
	ldr	q1, [x11]
	subs	x9, x9, 1
	prfm	pldl1keep, [x7, 0x40]
	add	x7, x7, x12
	prfm	pldl1keep, [x11,0x40]
	add	x11,x11,x12
	stp	q0, q1, [x2], 0x20
	b.ne	channel_loop2

channel_last:
	cbz	x10, channel_loop_end	
	ldr	q0, [x7]
	prfm	pldl1keep, [x7, 0x40]
	str	q0, [x2], 0x10

channel_loop_end:

	add	x6, x6, 16
	subs	x3, x3, 1
	b.ne	col_loop

col_end:

	ret
	.end
