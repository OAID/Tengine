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
 * Copyright (c) 2019, Open AI Lab
 * Author: xiaowei@openailab.com, chunyinglv@openailab.com
 */

// x0        output         v12-v15
// x1        input          v8,v9
// x2        kernel         v0,v1
// x3        cin

// x4       cin/4
// x3       cin_resi_4

    .section .text, "ax"
    .align 5

    .type wino_sgemm_4x4_A7 STT_FUNC
    .global wino_sgemm_4x4_A7
    .hidden wino_sgemm_4x4_A7

wino_sgemm_4x4_A7:
    push        {r4, lr}

    vmov.i64    q12, #0x0
    vmov.i64    q13, #0x0
    vmov.i64    q14, #0x0
    vmov.i64    q15, #0x0

    cmp        r3, #0x4
    blt        loop4_end
    lsr        r4, r3, #0x2

// x0        output         v12-v15
// x1        input          v8,v9
// x2        kernel         v0,v1
loop4:
    vldr        d0,  [r2]
    vldm		r1!, {d16-d19}
    vldr        d1,  [r2, #0x8]
	vldr		d2,[r2, #0x10]		// k11, k01
	vldr		d3,[r2, #0x18]		// k31, k21   
    

    vmla.f32    d24, d16, d0[0]
    vmla.f32    d25, d17, d0[0]
    vmla.f32    d26, d16, d0[1]
    vmla.f32    d27, d17, d0[1]
    vmla.f32    d28, d16, d1[0]
    vmla.f32    d29, d17, d1[0]
    vmla.f32    d30, d16, d1[1]
    vmla.f32    d31, d17, d1[1]

    vmla.f32    d24, d18, d2[0]
    vmla.f32    d25, d19, d2[0]
    vmla.f32    d26, d18, d2[1]
    vmla.f32    d27, d19, d2[1]
    vmla.f32    d28, d18, d3[0]
    vmla.f32    d29, d19, d3[0]
    vmla.f32    d30, d18, d3[1]
    vmla.f32    d31, d19, d3[1]

	pld		[r1, #0x220]
	pld		[r2, #0x240]
	vldr		d0,[r2, #0x20]		// k12, k02
	vldm		r1!, {d16-d19}		// i[3-0][3-2]
	vldr		d1,[r2, #0x28]		// k32, k22
	vldr		d2,[r2, #0x30]		// k13, k03
	vldr		d3,[r2, #0x38]		// k33, k23
	add		r2, r2, #0x40

    vmla.f32    d24, d16, d0[0]
    vmla.f32    d25, d17, d0[0]
    vmla.f32    d26, d16, d0[1]
    vmla.f32    d27, d17, d0[1]
    vmla.f32    d28, d16, d1[0]
    vmla.f32    d29, d17, d1[0]
    vmla.f32    d30, d16, d1[1]
    vmla.f32    d31, d17, d1[1]

    vmla.f32    d24, d18, d2[0]
    vmla.f32    d25, d19, d2[0]
    vmla.f32    d26, d18, d2[1]
    vmla.f32    d27, d19, d2[1]
    vmla.f32    d28, d18, d3[0]
    vmla.f32    d29, d19, d3[0]
    vmla.f32    d30, d18, d3[1]
    vmla.f32    d31, d19, d3[1]

    subs        r4, r4, #0x1
    bne        loop4

loop4_end:
    ands        r3, r3, #0x3
    beq        save_result

loop1:
    vldm        r1!, {d16 - d17}    // i[3-0]0
    vldm        r2!, {d0  -  d1}    // k[3-0]0
    subs        r3, r3, #0x1

    vmla.f32    d24, d16, d0[0]
    vmla.f32    d25, d17, d0[0]
    vmla.f32    d26, d16, d0[1]
    vmla.f32    d27, d17, d0[1]
    vmla.f32    d28, d16, d1[0]
    vmla.f32    d29, d17, d1[0]
    vmla.f32    d30, d16, d1[1]
    vmla.f32    d31, d17, d1[1]


    bne        loop1

save_result:
    vstm        r0, {d24-d31}

end:
    pop        {r4,pc}

    .end
