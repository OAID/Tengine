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
 * Author: ddzhao@openailab.com
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
  .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff

im2col_fp32_3x3:
        addi            sp, sp, -56
        sd              t0, 0(sp)
        sd              t1, 8(sp)
        sd              t2, 16(sp)
        sd              t3, 24(sp)
        sd              t4, 32(sp)
        sd              t5, 40(sp)
        sd              t6, 48(sp)
        vsetvli         t0, a0, e32
	// initial
        beqz            a3, finish
        li              t0, 2
        slli	        a1, a1, 2
        mul             a2, a2, a1
        add             t5, a0, a1
        slli	        t1, a1, 1
        add             t6, a0, t1
        li              t2, 8
        beq             a5, t0, stride2_channel_loop

stride1_channel_loop:
        vlw.v           v0, (a0)
        addi            t0, a0, 16
        vlw.v           v1, (t0)
        vlw.v           v2, (t5)
        addi            t0, t5, 16
        vlw.v           v3, (t0)
        vlw.v           v4, (t6)
        addi            t0, t6, 16
        vlw.v           v5, (t0)
        
        addi             a3, a3, -1
        
        addi            t0, a0, 4
        vlw.v           v16, (t0)
        addi            t0, a0, 8
        vlw.v           v17, (t0)
        add             a0, a0, a2
        
        addi            t0, t5, 4
        vlw.v           v19, (t0)
        
        addi            t0, t5, 8
        vlw.v           v20, (t0)
        add             t5, t5, a2
        addi            t0, t6, 4
        vlw.v           v22, (t0)
        addi            t0, t6, 8
        vlw.v           v23, (t0)
        add             t6, t6, a2
        vsw.v           v0, (a4)
        addi            a4, a4, 16
        vsw.v           v16, (a4)
        addi            a4, a4, 16
        vsw.v           v17, (a4)
        addi            a4, a4, 16
        vsw.v           v2, (a4)
        addi            a4, a4, 16
        vsw.v           v19, (a4)
        addi            a4, a4, 16
        vsw.v           v20, (a4)
        addi            a4, a4, 16
        vsw.v           v4, (a4)
        addi            a4, a4, 16
        vsw.v           v22, (a4)
        addi            a4, a4, 16
        vsw.v           v23, (a4)
        addi            a4, a4, 16
        bnez            a3, stride1_channel_loop
        j               finish

stride2_channel_loop:
        la              t0, mask_32b
        vlw.v           v0, (t0)
        addi            t0, a0, 0
        vlsw.v          v16, (t0), t2
        addi            t0, a0, 0x4
        vlsw.v          v17, (t0), t2
        addi            t0, a0, 32
        vlw.v           v18, (t0)
        vslidedown.vi   v1, v16, 1
        vslideup.vi     v2, v18, 3
        vmerge.vvm      v18, v1, v2, v0
        
        addi            t0, t5, 0
        vlsw.v           v19, (t0), t2
        addi            t0, t5, 0x4
        vlsw.v           v20, (t0), t2
        addi            t0, t5, 0x20
        vlw.v           v21, (t0)
        vslidedown.vi   v1, v19, 1
        vslideup.vi     v2, v21, 3
        vmerge.vvm      v21, v1, v2, v0
        
        addi            t0, t6, 0
        vlsw.v           v22, (t0), t2
        addi            t0, t6, 0x4
        vlsw.v           v23, (t0), t2
        addi            t0, t6, 0x20
        vlw.v           v24, (t0)
        vslidedown.vi   v1, v22, 1
        vslideup.vi     v2, v24, 3
        vmerge.vvm      v24, v1, v2, v0
        
        addi            a3, a3, -1
        
        vsw.v           v16, (a4)
        addi            a4, a4, 0x10
        vsw.v           v17, (a4)
        addi            a4, a4, 0x10
        vsw.v           v18, (a4)
        addi            a4, a4, 0x10
        vsw.v           v19, (a4)
        addi            a4, a4, 0x10
        vsw.v           v20, (a4)
        addi            a4, a4, 0x10
        vsw.v           v21, (a4)
        addi            a4, a4, 0x10
        vsw.v           v22, (a4)
        addi            a4, a4, 0x10
        vsw.v           v23, (a4)
        addi            a4, a4, 0x10
        vsw.v           v24, (a4)
        addi            a4, a4, 0x10
        
	add	        a0, a0, a2
        add	        t5, t5, a2
        add	        t6, t6, a2
        
        bnez            a3, stride2_channel_loop
finish:
        ld              t0, 0(sp)
        ld              t1, 8(sp)
        ld              t2, 16(sp)
        ld              t3, 24(sp)
        ld              t4, 32(sp)
        ld              t5, 40(sp)
        ld              t6, 48(sp)
        addi            sp, sp, 56
	ret
	.end
