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

// x0        output         v13-v15
// x1        input          v0
// x2        kernel         v1-v3
// x3        cin

// x4       cin/4
// x3       cin_resi_4

// loop4:12 fmla
// r1: 0x10
// r2: 0xc0 = 3 0x40

    .section .text, "ax"
    .align 5

    .type wino_sgemm_1x12_A17 STT_FUNC
    .global wino_sgemm_1x12_A17
    .hidden wino_sgemm_1x12_A17

wino_sgemm_1x12_A17:
    push        {r4, lr}

    vmov.i64    q13, #0x0
    vmov.i64    q14, #0x0
    vmov.i64    q15, #0x0

    cmp        r3, #0x4
    blt        loop4_end
    lsr        r4, r3, #0x2

loop4:
    vldm        r1!,{d0-d1}       //q0
    vldm        r2,{d2-d7}        //q1,q2,q3
    subs        r4, r4, #1

    vmla.f32    q13,q1, d0[0]
    vmla.f32    q14,q2, d0[0]

    vldr        d2,[r2,#0x30]
    vldr        d3,[r2,#0x38]
    vmla.f32    q15,q3, d0[0]    //d0[0]

    vldr        d4,[r2,#0x40]
    vldr        d5,[r2,#0x48]
    vmla.f32    q13,q1, d0[1]
    
    vldr        d6,[r2,#0x50]
    vldr        d7,[r2,#0x58]
    vmla.f32    q14,q2, d0[1]

    vldr        d2,[r2,#0x60]
    vldr        d3,[r2,#0x68]
    vmla.f32    q15,q3, d0[1]   //d0[1]

    vldr        d4,[r2,#0x70]
    vldr        d5,[r2,#0x78]
    vmla.f32    q13,q1, d1[0]

    vldr        d6,[r2,#0x80]
    vldr        d7,[r2,#0x88]
    vmla.f32    q14,q2, d1[0]
    pld        [r2,#0x240]
    vldr        d2,[r2,#0x90]
    vldr        d3,[r2,#0x98]
    vmla.f32    q15,q3, d1[0]   //d1[0]
    pld        [r2,#0x280]
    vldr        d4,[r2,#0xa0]
    vldr        d5,[r2,#0xa8]
    vmla.f32    q13,q1, d1[1]
    pld        [r2,#0x2c0]
    vldr        d6,[r2,#0xb0]
    vldr        d7,[r2,#0xb8]
    add        r2, r2, #0xc0
    vmla.f32    q14,q2, d1[1]
    vmla.f32    q15,q3, d1[1]   //d1[1]

    bne        loop4

loop4_end:
    ands        r3, r3, #0x3
    beq        save_result

loop1:
    vldr        s0,[r1]
    vldm        r2!,{d2-d7}        // k[11-0][0]
    vmla.f32    q13,q1, d0[0]
    subs        r3, r3, #0x1
    vmla.f32    q14,q2, d0[0]
    add         r1, r1, #0x4
    vmla.f32    q15,q3, d0[0]
    bne        loop1

save_result:
    vstm        r0, {d26-d31}

end:
    pop        {r4,pc}

    .end
