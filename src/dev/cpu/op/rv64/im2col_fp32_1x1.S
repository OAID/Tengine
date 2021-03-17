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
	vsetvli	t0, a0, e64, m1, d1
# 	cmp	x3, 4
# 	b.lt	col_end
	li 		t0, 4
	blt 	a3, t0, col_end
# 	lsr	x3, x3, 2	// x3 = col_cnt
	srli	a3, a3, 2
# 	lsl	x1, x1, 2	// x1 = input_xy size
	slli	a1, a1, 2
# 	mov	x6, x0										// x6  -> t6
	mv 		t6, a0
# 	lsl	x12,x1, 1	// x12 = input_xy size * 2		// x12 -> t5
	slli	t5, a1, 1
# 	and	x10,x4, 1
	add 	t4, a4, 1								// x10 -> t4

	// col loop
col_loop:
# 	mov	x7, x6		// x7 = input					// x7  -> t3
	mv 		t3, t6
# 	lsr	x9, x4, 1	// x9 = channel cnt				// x9  -> t2
	srli	t2, a4, 1
# 	cbz	x9, channel_last
	beqz	t2, channel_last
# 	add	x11,x7, x1									// x11 -> t1
	add 	t1, t3, a1						
# 	// kernel size loop
channel_loop2:
# 	ldr	q0, [x7]
	vlb.v 	v0,(t3)
# 	ldr	q1, [x11]
	vlb.v 	v1,(t1)
# 	subs	x9, x9, 1
	addi 	t2, t2, -1
# 	prfm	pldl1keep, [x7, 0x40]
# 	add		x7, x7, x12
	add 	t3, t3, t5
# 	prfm	pldl1keep, [x11,0x40]
# 	add		x11,x11,x12
	add 	t1, t1, t5
# 	stp	q0, q1, [x2], 0x20
	vsb.v 	v0, (a2)
	addi 	a2, a2, 16
	vsb.v 	v1, (a2)
	addi 	a2, a2, 16
# 	b.ne	channel_loop2
	bnez	t2, channel_loop2

channel_last:
# 	cbz	x10, channel_loop_end	
	beqz 	t4, channel_loop_end
# 	ldr	q0, [x7]
	vlb.v 	v0,(t3)
# 	prfm	pldl1keep, [x7, 0x40]
# 	str	q0, [x2], 0x10
	vsb.v 	v0, (a2)
	addi 	a2, a2, 16

channel_loop_end:
# 	add	x6, x6, 16
	addi 	t6, t6, 16
# 	subs	x3, x3, 1
	addi 	a3, a3, -1
# 	b.ne	col_loop
	bnez	a3, col_loop

col_end:

	ret
	.end
