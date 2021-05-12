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

// x0        output         v4-v15
// x1        input          v0
// x2        kernel         v1-v3
// x3        cin

// x4       cin/4
// x3       cin_resi_4

    .section .text, "ax"
    .align 5

    .type wino_sgemm_4x12_A17 STT_FUNC
    .global wino_sgemm_4x12_A17
    .hidden wino_sgemm_4x12_A17

wino_sgemm_4x12_A17:
    pld        [r1,#0x80]
    push        {r4, lr}
    vpush        {d8-d15}

    vmov.i64    q4,  #0x0
    vmov.i64    q5,  #0x0
    vmov.i64    q6,  #0x0
    vmov.i64    q7,  #0x0
    vmov.i64    q8,  #0x0
    vmov.i64    q9,  #0x0
    vmov.i64    q10, #0x0
    vmov.i64    q11, #0x0
    vmov.i64    q12, #0x0
    vmov.i64    q13, #0x0
    vmov.i64    q14, #0x0
    vmov.i64    q15, #0x0

    cmp        r3, #0x4
    blt        loop4_end
    lsr        r4, r3, #0x2

loop4:
    vldm        r1!,{d0-d1}        // i[3-0][0]
    vldm        r2,{d2-d7}        // k[11-0][0]
    subs        r4, r4, #1
    vmla.f32    q4, q0, d2[0]
    vmla.f32    q5, q0, d2[1]
    vmla.f32    q6, q0, d3[0]
    vmla.f32    q7, q0, d3[1]
    vmla.f32    q8, q0, d4[0]
    vmla.f32    q9, q0, d4[1]
    vldr        d2,[r2,#0x30]
    vldr        d3,[r2,#0x38]
    vmla.f32    q10,q0, d5[0]
    vmla.f32    q11,q0, d5[1]
    vmla.f32    q12,q0, d6[0]
    vmla.f32    q13,q0, d6[1]
    vmla.f32    q14,q0, d7[0]
    vmla.f32    q15,q0, d7[1]
    vldm        r1!,{d0-d1}        // i[3-0][0]
    vldr        d4,[r2,#0x40]
    vldr        d5,[r2,#0x48]
    vmla.f32    q4, q0, d2[0]
    vmla.f32    q5, q0, d2[1]
    vmla.f32    q6, q0, d3[0]
    vmla.f32    q7, q0, d3[1]
    vldr        d6,[r2,#0x50]
    vldr        d7,[r2,#0x58]
    vmla.f32    q8, q0, d4[0]
    vmla.f32    q9, q0, d4[1]
    vmla.f32    q10,q0, d5[0]
    vmla.f32    q11,q0, d5[1]
    vldr        d2,[r2,#0x60]
    vldr        d3,[r2,#0x68]
    vmla.f32    q12,q0, d6[0]
    vmla.f32    q13,q0, d6[1]
    vmla.f32    q14,q0, d7[0]
    vmla.f32    q15,q0, d7[1]
    vldm        r1!,{d0-d1}        // i[3-0][0]
    vldr        d4,[r2,#0x70]
    vldr        d5,[r2,#0x78]
    vmla.f32    q4, q0, d2[0]
    vmla.f32    q5, q0, d2[1]
    vmla.f32    q6, q0, d3[0]
    vmla.f32    q7, q0, d3[1]
    vldr        d6,[r2,#0x80]
    vldr        d7,[r2,#0x88]
    vmla.f32    q8, q0, d4[0]
    vmla.f32    q9, q0, d4[1]
    vmla.f32    q10,q0, d5[0]
    vmla.f32    q11,q0, d5[1]
    vldr        d2,[r2,#0x90]
    vldr        d3,[r2,#0x98]
    vmla.f32    q12,q0, d6[0]
    vmla.f32    q13,q0, d6[1]
    vmla.f32    q14,q0, d7[0]
    vmla.f32    q15,q0, d7[1]
    vldm        r1!,{d0-d1}        // i[3-0][0]
    vldr        d4,[r2,#0xa0]
    vldr        d5,[r2,#0xa8]
    vmla.f32    q4, q0, d2[0]
    vmla.f32    q5, q0, d2[1]
    pld        [r1,#0x140]
    vmla.f32    q6, q0, d3[0]
    vmla.f32    q7, q0, d3[1]
    vldr        d6,[r2,#0xb0]
    vldr        d7,[r2,#0xb8]
    vmla.f32    q8, q0, d4[0]
    vmla.f32    q9, q0, d4[1]
    pld        [r2,#0x380]
    vmla.f32    q10,q0, d5[0]
    pld        [r2,#0x3c0]
    vmla.f32    q11,q0, d5[1]
    pld        [r2,#0x400]
    vmla.f32    q12,q0, d6[0]
    add        r2, r2, #0xc0
    vmla.f32    q13,q0, d6[1]
    vmla.f32    q14,q0, d7[0]
    vmla.f32    q15,q0, d7[1]
    bne        loop4

loop4_end:
    ands        r3, r3, #0x3
    beq        save_result

loop1:
    vldm        r1!,{d0-d1}        // i[3-0][0]
    vldm        r2!,{d2-d7}        // k[11-0][0]
    vmla.f32    q4, q0, d2[0]
    vmla.f32    q5, q0, d2[1]
    vmla.f32    q6, q0, d3[0]
    vmla.f32    q7, q0, d3[1]
    vmla.f32    q8, q0, d4[0]
    vmla.f32    q9, q0, d4[1]
    vmla.f32    q10,q0, d5[0]
    vmla.f32    q11,q0, d5[1]
    vmla.f32    q12,q0, d6[0]
    vmla.f32    q13,q0, d6[1]
    vmla.f32    q14,q0, d7[0]
    vmla.f32    q15,q0, d7[1]
    subs        r3, r3, #0x1
    bne        loop1

save_result:
    vstm        r0!, {d8-d23}
    vstm        r0, {d24-d31}

end:
    vpop       {d8-d15}
    pop        {r4,pc}

    .end
