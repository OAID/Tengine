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
/* relu implementation using neon vector */


.text
.align 5
.global relu_neon
.type   relu_neon, %function

relu_neon:

   //x0 addr
   //x1 float number

   fmov s0,wzr
   dup v31.4s,v0.s[0] //initial the compare vector
   mov x4,x0   //x4 is the store buffer

   lsr x3,x1,8  // 256
   cbz x3,less_256
   
   //initial load
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64

   cmp  x3,0x2
   and  x1,x1,0xff
   blt  last_256
   sub  x3,x3,1  //saved for the last_256
   
loop_256:      
   
   fmax v0.4s,v31.4s,v0.4s   //block 0
   fmax v1.4s,v31.4s,v1.4s

   prfm	pldl1keep, [x0,0x0c0] 
   prfm	pldl2keep, [x0,0x200] 

   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s

 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s   //block 1
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 2
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 3
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s
 
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s  //block 4
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s //block 5
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s //block 6
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 7
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s

   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s   //block 8
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s  //block 9
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 10
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 11
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s
 
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s  //block 12
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s //block 13
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 14
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   subs x3,x3,1
   
   fmax v12.4s,v31.4s,v12.4s  //block 15
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0xc0] 
#   prfm	pldl2keep, [x0,0x200] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s

   
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64

   b.ne  loop_256
   
last_256:
   
   fmax v0.4s,v31.4s,v0.4s   //block 0
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x200] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s   //block 1
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 2
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 3
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s
 
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s  //block 4
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s //block 5
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s //block 6
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 7
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s

   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s   //block 8
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s  //block 9
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 10
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 11
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s
 
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
 
   fmax v0.4s,v31.4s,v0.4s  //block 12
   fmax v1.4s,v31.4s,v1.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s //block 13
   fmax v5.4s,v31.4s,v5.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 14
   fmax v9.4s,v31.4s,v9.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s  //block 15
   fmax v13.4s,v31.4s,v13.4s
   prfm	pldl1keep, [x0,0x0c0] 
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s

   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
   
less_256:   
   
   //if more than 128?
   cmp x1,128
   blt  less_128
   and  x1,x1,0x7f
 
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   
   fmax v0.4s,v31.4s,v0.4s   //block 0
   fmax v1.4s,v31.4s,v1.4s
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s   //block 1
   fmax v5.4s,v31.4s,v5.4s
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s  //block 2
   fmax v9.4s,v31.4s,v9.4s
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x0],#64
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 3
   fmax v13.4s,v31.4s,v13.4s
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s
 
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x0],#64
   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64

   fmax v0.4s,v31.4s,v0.4s  //block 4
   fmax v1.4s,v31.4s,v1.4s
   fmax v2.4s,v31.4s,v2.4s
   fmax v3.4s,v31.4s,v3.4s
 
   ld1 {v8.4s,v9.4s,v10.4s,v11.4s},[x0],#64
   st1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4],#64

   fmax v4.4s,v31.4s,v4.4s //block 5
   fmax v5.4s,v31.4s,v5.4s
   fmax v6.4s,v31.4s,v6.4s
   fmax v7.4s,v31.4s,v7.4s
   
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   st1 {v4.4s,v5.4s,v6.4s,v7.4s},[x4],#64
   
   fmax v8.4s,v31.4s,v8.4s //block 6
   fmax v9.4s,v31.4s,v9.4s
   fmax v10.4s,v31.4s,v10.4s
   fmax v11.4s,v31.4s,v11.4s
   
   st1 {v8.4s,v9.4s,v10.4s,v11.4s},[x4],#64
   
   fmax v12.4s,v31.4s,v12.4s //block 7
   fmax v13.4s,v31.4s,v13.4s
   fmax v14.4s,v31.4s,v14.4s
   fmax v15.4s,v31.4s,v15.4s

   st1 {v12.4s,v13.4s,v14.4s,v15.4s},[x4],#64
   

less_128:

   lsr x3,x1,2  // 4
   cbz x3,less_4
   and x1,x1,0x3
   
loop_4:
   ld1 {v0.4s},[x0],#16
   subs x3,x3,1
   fmax v0.4s,v0.4s,v31.4s
   st1 {v0.4s},[x4],#16
   b.ne  loop_4

less_4:
   cbz x1, done
   
loop_1:
   ldr s0,[x0],#4
   fmax s0,s0,s31
   str s0,[x4],#4
   subs x1,x1,1
   b.ne loop_1
   
done:

   ret
   

.end


   
