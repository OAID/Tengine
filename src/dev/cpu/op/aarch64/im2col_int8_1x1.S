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
//         x4 arg4  kernel_size
//         x5 arg5  scale address
//
// register definition
//    x0 input address 
//    x1 input_xy x 1
//    x2 col address
//    x3 col_cnt
//    x4 kernel_size
//    x5 input pointer
//    x6 channel cnt
//    x7 input address + input_xy
//    x9 
//    x10

        .section .text,"ax"
        .align 5

        .type   im2col_int8_1x1 STT_FUNC
        .global im2col_int8_1x1
        .hidden im2col_int8_1x1
im2col_int8_1x1:
	cmp	    x3, 4
	b.lt	col_end
	lsr	    x3, x3, 2	// x3 = col_cnt

// col loop
col_loop:
	mov	    x5, x0		// x5 = input
	lsr	    x6, x4, 1	// x6 = channel cnt / 4
	add	    x7, x5, x1
	cbz	    x6, kernel_last1

// kernel size loop
kernel_loop2:
	ldr	    d0, [x5]
	ldr	    d1, [x7]
	subs	x6, x6, 1
    
	prfm	pldl1keep, [x5, 0x40]
    add     x5, x5, x1, lsl 1
	prfm	pldl1keep, [x7, 0x40]
    add     x7, x7, x1, lsl 1
    
	zip1	v2.8b, v0.8b, v1.8b
    
	st1	    {v2.8b}, [x2]
    
    add     x2, x2, 8
    b.ne    kernel_loop2

kernel_last1:
    and     x9, x4, 0x1
	cbz	    x9, kernel_loop_end

	// last channel
	ldr	    d0, [x5]
	movi	d1, 0
	zip1	v2.8b, v0.8b, v1.8b
    
	st1	    {v2.8b}, [x2]
    add     x2, x2, 8

kernel_loop_end:
	add	    x0, x0, 4
	subs	x3, x3, 1
	b.ne	col_loop

col_end:

	ret
	.end
