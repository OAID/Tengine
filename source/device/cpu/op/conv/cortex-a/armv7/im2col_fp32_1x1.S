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
//         r0 arg0  input address 
//         r1 arg1  input_xy
//         r2 arg2  col address
//         r3 arg3  col_cnt must be multiply of 4
//         sp arg4  input channel
//
// register definition
//    r0 input address 
//    r1 input_xy x 4
//    r2 col address
//    r3 col_cnt
//    r4 channel cnt
//    r5 input pointer
//    r6 input pointer + input_xy

        .section .text,"ax"
        .align 5

        .type   im2col_fp32_1x1 STT_FUNC
        .global im2col_fp32_1x1
        .hidden im2col_fp32_1x1
im2col_fp32_1x1:
	push	{r4-r6,lr}
	cmp	r3, #4
	blt	col_end
	lsr	r3, r3, #2	// r3 = col_cnt
	lsl	r1, r1, #2	// r1 = input_xy size

	// col loop
col_loop:
	mov	r5, r0		// r5 = input
	ldr	r4, [sp, #0x10]
	lsr	r4, r4, #1	// r4 = channel cnt
	cmp	r4, #0
	beq	channel_last
	add	r6, r5, r1
	// kernel size loop
channel_loop2:
	vldr	d0, [r5]
	vldr	d1, [r5, #0x8]
	subs	r4, r4, #1
	vldr	d2, [r6]
	vldr	d3, [r6, #0x8]
	pld	[r5, #0x40]
	add	r5, r5, r1, LSL #1
	pld	[r6, #0x40]
	add	r6, r6, r1, LSL #1
	vstm	r2!, {d0-d3}
	bne	channel_loop2

channel_last:
	ldr	r4, [sp, #0x10]
	tst	r4, #1
	beq	channel_loop_end	

	vldm	r5, {d0-d1}
	pld	[r5, #0x40]
	vstm	r2!,{d0-d1}

channel_loop_end:
	add	r0, r0, #16
	subs	r3, r3, #1
	bne	col_loop

col_end:

	pop	{r4-r6,pc}
	.end
