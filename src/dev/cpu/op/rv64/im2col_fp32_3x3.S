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
// im2col fp16 for kernel 3x3  include 2 function  stride 1 and stride 2
// ABCDABCD
//
// input:
//         x0 arg0  input address 
//         x1 arg1  input_x
//         x2 arg2  input_y
//         x3 arg3  input channel cnt
//         x4 arg4  col address
//         x5 arg5  stride_x
//
// register definition
//    x0 cl0 address  q0  q1    d16 d17 d18
//    x1 input_x x 4
//    x2 input_xy x 4
//    x3 input channel
//    x4 col address
//    x5 stride_x
//    x11 cl1 address q2  q3    d19 d20 d21
//    x12 cl2 address q4  q5    d22 d23 d24

        .section .text,"ax"
        .align 5

        .type   im2col_fp32_3x3 STT_FUNC
        .global im2col_fp32_3x3
        .hidden im2col_fp32_3x3

.balign 16
mask_32b:
  .byte 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
        0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01

im2col_fp32_3x3:
        vsetvli         t0, a0, e32
	// initial
# 	cbz	        x3, finish
        beqz            a3, finish
# 	cmp	        x5, 2
        li              t0, 2
# 	lsl	        x1, x1, 2	// x1 = input_x size
        slli	        a1, a1, 2
# 	mul	        x2, x2, x1	// x2 = input_xy size
        mul             a2, a2, a1
# 	add	        x11,x0, x1                      // x11 -> t5
        add             t5, a0, a1
# 	add	        x12,x0, x1, LSL 1               // x12 -> t6
        slli	        t1, a1, 1
        add             t6, a0, t1
# 	beq	        stride2_channel_loop
        beq             a5, t0, stride2_channel_loop

stride1_channel_loop:
# 	ldr	        q0,  [x0]
# 	ldr	        d1,  [x0, 0x10]	
        vlw.v           v0, (a0)
        addi            t0, a0, 16
        vlw.v           v1, (t0)
# 	ldr	        q2,  [x11]
# 	ldr	        d3,  [x11,0x10]	
        vlw.v           v2, (t5)
        addi            t0, t5, 16
        vlw.v           v3, (t0)
# 	ldr	        q4,  [x12]
# 	ldr	        d5,  [x12,0x10]	
        vlw.v           v4, (t6)
        addi            t0, t6, 16
        vlw.v           v5, (t0)
# 	subs	        x3, x3, 1
        addi             a3, a3, -1
# 	ext	        v16.16b, v0.16b, v1.16b, 4
        addi            t0, a0, 4
        vlw.v           v16, (t0)
# 	prfm	        pldl1strm, [x0, 0x40]
# 	ext	        v17.16b, v0.16b, v1.16b, 8
        addi            t0, a0, 8
        vlw.v           v17, (t0)
# 	add	        x0, x0, x2
        add             a0, a0, a2
# 	ext	        v19.16b, v2.16b, v3.16b, 4
        addi            t0, t5, 4
        vlw.v           v19, (t0)
# 	prfm	        pldl1strm, [x11,0x40]
# 	ext	        v20.16b, v2.16b, v3.16b, 8
        addi            t0, t5, 8
        vlw.v           v20, (t0)
# 	add	        x11,x11,x2
        add             t5, t5, a2
# 	ext	        v22.16b, v4.16b, v5.16b, 4
        addi            t0, t6, 4
        vlw.v           v22, (t0)
# 	prfm	        pldl1strm, [x12,0x40]
# 	ext	        v23.16b, v4.16b, v5.16b, 8
        addi            t0, t6, 8
        vlw.v           v23, (t0)
# 	add	        x12,x12,x2
        add             t6, t6, a2
# 	stp	        q0, q16, [x4], 0x20
        vsw.v           v0, (a4)
        addi            a4, a4, 16
        vsw.v           v16, (a4)
        addi            a4, a4, 16
# 	stp	        q17,q2,  [x4], 0x20
        vsw.v           v17, (a4)
        addi            a4, a4, 16
        vsw.v           v2, (a4)
        addi            a4, a4, 16
# 	stp	        q19,q20, [x4], 0x20
        vsw.v           v19, (a4)
        addi            a4, a4, 16
        vsw.v           v20, (a4)
        addi            a4, a4, 16
# 	stp	        q4, q22, [x4], 0x20
        vsw.v           v4, (a4)
        addi            a4, a4, 16
        vsw.v           v22, (a4)
        addi            a4, a4, 16
# 	str	        q23, [x4], 0x10
        vsw.v           v23, (a4)
        addi            a4, a4, 16
# 	bne	        stride1_channel_loop
        bnez            a3, stride1_channel_loop
# 	b	        finish
        j               finish

stride2_channel_loop:
        la              t0, mask_32b
        vlw.v           v0, (t0)
# 	ld2	        {v16.4s, v17.4s}, [x0]
# 	ldr	        s18, [x0, 0x20]
# 	ext	        v18.16b,v16.16b, v18.16b, 4
        addi            t0, a0, 0
        vlw.v           v16, (t0)
        addi            t0, a0, 16
        vlw.v           v17, (t0)
        addi            t0, a0, 32
        vlw.v           v18, (t0)
        vslidedown.vi   v1, v16, 4
        vslideup.vi     v18, v18, 4
        vmerge.vvm      v18, v1, v18, v0
# 	ld2	        {v19.4s, v20.4s}, [x11]
# 	ldr	        s21, [x11,0x20]	
# 	ext	        v21.16b,v19.16b, v21.16b, 4
        addi            t0, t5, 0
        vlw.v           v19, (t0)
        addi            t0, t5, 16
        vlw.v           v20, (t0)
        addi            t0, t5, 32
        vlw.v           v21, (t0)
        vslidedown.vi   v1, v19, 4
        vslideup.vi     v21, v21, 4
        vmerge.vvm      v21, v1, v21, v0
# 	ld2	        {v22.4s, v23.4s}, [x12]
# 	ldr	        s24, [x12,0x20]	
# 	ext	        v24.16b,v22.16b, v24.16b, 4
        addi            t0, t5, 0
        vlw.v           v19, (t0)
        addi            t0, t5, 16
        vlw.v           v20, (t0)
        addi            t0, t5, 32
        vlw.v           v21, (t0)
        vslidedown.vi   v1, v19, 4
        vslideup.vi     v21, v21, 4
        vmerge.vvm      v21, v1, v21, v0
# 	subs	        x3, x3, 1
        addi            a3, a3, -1
# 	prfm	        pldl1strm, [x0, 0x60]
# 	prfm	        pldl1strm, [x11,0x60]
# 	prfm	        pldl1strm, [x12,0x60]
        vsw.v           v16, (a4)
        addi            a4, a4, 16
        vsw.v           v17, (a4)
        addi            a4, a4, 16
        vsw.v           v18, (a4)
        addi            a4, a4, 16
        vsw.v           v19, (a4)
        addi            a4, a4, 16
        vsw.v           v20, (a4)
        addi            a4, a4, 16
        vsw.v           v21, (a4)
        addi            a4, a4, 16
        vsw.v           v22, (a4)
        addi            a4, a4, 16
        vsw.v           v23, (a4)
        addi            a4, a4, 16
        vsw.v           v24, (a4)
        addi            a4, a4, 16
        
	add	        a0, a0, a2
        add	        t5, t5, a2
        add	        t6, t6, a2
# 	stp	        q16, q17, [x4], 0x20
# 	add	        x0, x0, x2
# 	stp	        q18, q19, [x4], 0x20
# 	add	        x11,x11,x2
# 	stp	        q20, q21, [x4], 0x20
# 	add	        x12,x12,x2
# 	stp	        q22, q23, [x4], 0x20
# 	str	        q24, [x4], 0x10
# 	bne	        stride2_channel_loop
        bnez            a3, stride2_channel_loop
finish:
	ret

	.end
