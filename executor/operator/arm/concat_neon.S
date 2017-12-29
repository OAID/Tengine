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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
.text
.align 5
.global concat_neon
.type   concat_neon, %function

//x0: output
//x1: input
//x2: data_size

concat_neon:
    lsr x8,x2,8
    lsl x9,x8,8    //64 float per loop
    sub x9,x2,x9
    cbz x8,less_64

concat_loop_start:

    ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x1],#64
    ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x1],#64

    st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
    ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x1],#64

    st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
    ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x1],#64

    subs x8,x8,1

    st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
    st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64

    b.ne concat_loop_start

less_64:

    cbz x9, all_done
    subs x9,x9,1

    ldr s0,[x1],#4
    str s0,[x0],#4

    b.ne less_64

all_done:

    ret

