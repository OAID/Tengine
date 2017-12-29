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
//x0: input
//x1: h
//x2: w
//x3: kernel
//x4: output //L-2

//x5: L-1 output
//x6: L0 output
//x7: processed item
//x8: counter
//x9: x2*4 

//v0-v3: L-2  
//v4-v7: L-1  
//v8-v11: L0  
//v12-v15: input 
//v18 -- post vector
//v24-v26: kernel
//v17 --- saved previous vector
// v28,v29 --- shifted 

.text
.align 5
.global dw_k3s1p1
.type dw_k3s1p1, %function


dw_k3s1p1:

   //Load Kernel
   ld1 {v24.4s,v25.4s,v26.4s}, [x3]

   ext  v26.16b,v25.16b,v26.16b,8
   ext  v25.16b,v24.16b,v25.16b,12

   lsl x9,x2,2
   fmov s31,wzr

first_row_start:
   sub x1,x1,1

   lsr  x8,x2,4  //16 elem per time 
   lsl  x7,x8,4 

   ins  v17.s[3],v31.s[0]   //pre_vector for input

   mov x5,x4      //L-1
   add x6,x5,x9   //L-0

   cbz  x8,first_less_16
   sub x8,x8,1
   cbz x8,first_last_16_block
   
first_row_loop:

   //load 4 float input
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   ld1r {v16.4s},[x0]  

   //block 0
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
   
   //L-1: k1 input
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   //block 1
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04

   fmul v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
   
   fmul v9.4s,v18.4s,v24.s[0]   //k00
   fmla v9.4s,v13.4s,v24.s[1]   //k01
   fmla v9.4s,v19.4s,v24.s[2]   //k02
   st1 {v9.4s},[x6],#16

   //block 2
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04

   fmul v6.4s,v28.4s,v25.s[0]    //k10, 
   fmla v6.4s,v14.4s,v25.s[1]    //k11,
   fmla v6.4s,v29.4s,v25.s[2]    //k12
   st1 {v6.4s},[x5],#16
   
   fmul v10.4s,v28.4s,v24.s[0]   //k00
   fmla v10.4s,v14.4s,v24.s[1]   //k01
   fmla v10.4s,v29.4s,v24.s[2]   //k02
   st1 {v10.4s},[x6],#16

   //block 3: v15
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04

   fmul v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
   
   fmul v11.4s,v18.4s,v24.s[0]   //k00
   fmla v11.4s,v15.4s,v24.s[1]   //k01
   fmla v11.4s,v19.4s,v24.s[2]   //k02
   st1 {v11.4s},[x6],#16

   //done
   ins v17.s[3],v15.s[3]  //save prev vector

   //next loop
   subs x8,x8,1
   b.ne first_row_loop
   
first_last_16_block:
   //load 4 float input
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   dup v16.4s,v31.s[0] 
   
   cmp x7,x2  
   bge skip_load1
   
   ld1r {v16.4s},[x0]  
   
skip_load1:

   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
   
   //L-1: k1 xinput
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   //next four: v13
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04

   fmul v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
   
   fmul v9.4s,v18.4s,v24.s[0]   //k00
   fmla v9.4s,v13.4s,v24.s[1]   //k01
   fmla v9.4s,v19.4s,v24.s[2]   //k02
   st1 {v9.4s},[x6],#16


   //next four: v14
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04

   fmul v6.4s,v28.4s,v25.s[0]  //k10, 
   fmla v6.4s,v14.4s,v25.s[1]  //k11,
   fmla v6.4s,v29.4s,v25.s[2]  //k12
   st1 {v6.4s},[x5],#16
   
   fmul v10.4s,v28.4s,v24.s[0]   //k00
   fmla v10.4s,v14.4s,v24.s[1]   //k01
   fmla v10.4s,v29.4s,v24.s[2]   //k02
   st1 {v10.4s},[x6],#16

   //next four: v15
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04

   fmul v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
   
   fmul v11.4s,v18.4s,v24.s[0]   //k00
   fmla v11.4s,v15.4s,v24.s[1]   //k01
   fmla v11.4s,v19.4s,v24.s[2]   //k02
   st1 {v11.4s},[x6],#16
  
   ins v17.s[3],v15.s[3]  //save prev vector

first_less_16:
    sub x8,x2,x7
    cbz x8, first_row_done
    lsr x8,x8,2
    add x7,x7,x8,LSL #2
    cbz x8, first_less_4
   
first_loop_4_start:
  
    //4  nodes
   ld1 {v12.4s},[x0],#16

   cmp x8,2
   bge do_load2
   cmp x7,x2
   ins v13.s[0],v31.s[0]
   bge skip_load2
do_load2:
   ld1r {v13.4s},[x0]
skip_load2:
   
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         
   
   //L-1: k1 xinput
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   ins v17.s[3],v12.s[3] 

   subs x8,x8,1
   b.ne first_loop_4_start
   
first_less_4:
   sub x8,x2,x7
   cmp x8,1
   blt   first_row_done

first_1_2_3:   
   dup v12.4s,v31.s[0]
   dup v13.4s,v31.s[0]
   
   //2 or 3 items
   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
   sub x7,x8,1
   cbz x7, first_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
   sub x7,x8,2
   
   cbz x7, first_left_load_done
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]

first_left_load_done:         

   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-1   
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   
   //save result: 1, 2 or 3
   ins v28.s[0],v4.s[0]
   str  s28,[x5],#4
   
   ins v28.s[0],v8.s[0]
   str s28,[x6],#4
   
   
   cmp x8, 2
   blt  first_row_done
   
  ins v28.s[0],v4.s[1]
   str s28,[x5],#4
   
   
   ins v28.s[0],v8.s[1]
   str s28,[x6],#4
   
   cmp x8,3
   blt first_row_done
   
   ins v28.s[0],v4.s[2]
   str s28,[x5]
   
   ins v28.s[0],v8.s[2]
   str s28,[x6]

first_row_done:

 
mid_row_start:

   sub x1,x1,1
   cbz x1, mid_row_done

   lsr  x8,x2,4
   lsl  x7,x8,4
   
   add x5,x4,x9    //L-1
   add x6,x5,x9   //L0

   dup v17.4s,v31.s[0]
     
   cbz x8,mid_less_16
   sub x8,x8,1
   cbz x8,mid_last_16_block
  
mid_loop_start:
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4]  //L-2
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x5]  //L-1
  
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   ld1r {v16.4s},[x0]

   //block 0: v12
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
  
   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v0.4s,v29.4s,v26.s[2]  //k22
   st1 {v0.4s},[x4],#16
  
  
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
  
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   //block 1: v13
  
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04

  //L-2 
  fmla v1.4s,v18.4s,v26.s[0]  //k20, 
  fmla v1.4s,v13.4s,v26.s[1]  //k21,
  fmla v1.4s,v19.4s,v26.s[2]  //k22
  st1 {v1.4s},[x4],#16
  
  
  //L-1   
   fmla v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
  
   
   //L0
   fmul v9.4s,v18.4s,v24.s[0]   //k00
   fmla v9.4s,v13.4s,v24.s[1]   //k01
   fmla v9.4s,v19.4s,v24.s[2]   //k02
   st1 {v9.4s},[x6],#16
   
   //block 2: v14
  
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04
  //L-2 
  fmla v2.4s,v28.4s,v26.s[0]  //k20, 
  fmla v2.4s,v14.4s,v26.s[1]  //k21,
  fmla v2.4s,v29.4s,v26.s[2]  //k22
  st1 {v2.4s},[x4],#16
  
  
  //L-1   
   fmla v6.4s,v28.4s,v25.s[0]  //k10, 
   fmla v6.4s,v14.4s,v25.s[1]  //k11,
   fmla v6.4s,v29.4s,v25.s[2]  //k12
   st1 {v6.4s},[x5],#16
  
   //L0
   fmul v10.4s,v28.4s,v24.s[0]   //k00
   fmla v10.4s,v14.4s,v24.s[1]   //k01
   fmla v10.4s,v29.4s,v24.s[2]   //k02
   st1 {v10.4s},[x6],#16

   //block 3: v15
  
  //L-2 
  fmla v3.4s,v18.4s,v26.s[0]  //k20, 
  fmla v3.4s,v15.4s,v26.s[1]  //k21,
  fmla v3.4s,v19.4s,v26.s[2]  //k22
  st1 {v3.4s},[x4],#16
  
  
  //L-1   
   fmla v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
  
   
   //L0
   fmul v11.4s,v18.4s,v24.s[0]   //k00
   fmla v11.4s,v15.4s,v24.s[1]   //k01
   fmla v11.4s,v19.4s,v24.s[2]   //k02
   st1 {v11.4s},[x6],#16
  
   ins v17.s[3],v15.s[3]

   //next loop
   subs x8,x8,1
   b.ne mid_loop_start

mid_last_16_block:

   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4]  //L-2
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x5]  //L-1
  
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   dup v16.4s,v31.s[0]

   cmp x7,x2
   bge skip_load3
   ld1r {v16.4s},[x0]
   
 skip_load3:
    //block 0: v12
   
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
  
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   //block 1: v13
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v1.4s,v18.4s,v26.s[0]  //k20, 
  fmla v1.4s,v13.4s,v26.s[1]  //k21,
  fmla v1.4s,v19.4s,v26.s[2]  //k22
  st1 {v1.4s},[x4],#16
  
  
  //L-1   
   fmla v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
  
   
   //L0
   fmul v9.4s,v18.4s,v24.s[0]   //k00
   fmla v9.4s,v13.4s,v24.s[1]   //k01
   fmla v9.4s,v19.4s,v24.s[2]   //k02
   st1 {v9.4s},[x6],#16
   
   //block 2: v14
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v2.4s,v28.4s,v26.s[0]  //k20, 
  fmla v2.4s,v14.4s,v26.s[1]  //k21,
  fmla v2.4s,v29.4s,v26.s[2]  //k22
  st1 {v2.4s},[x4],#16
  
  
  //L-1   
   fmla v6.4s,v28.4s,v25.s[0]  //k10, 
   fmla v6.4s,v14.4s,v25.s[1]  //k11,
   fmla v6.4s,v29.4s,v25.s[2]  //k12
   st1 {v6.4s},[x5],#16
  
   
   //L0
   fmul v10.4s,v28.4s,v24.s[0]   //k00
   fmla v10.4s,v14.4s,v24.s[1]   //k01
   fmla v10.4s,v29.4s,v24.s[2]   //k02
   st1 {v10.4s},[x6],#16

   //block 3: v15
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v3.4s,v18.4s,v26.s[0]  //k20, 
  fmla v3.4s,v15.4s,v26.s[1]  //k21,
  fmla v3.4s,v19.4s,v26.s[2]  //k22
  st1 {v3.4s},[x4],#16
  
  
  //L-1   
   fmla v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
  
   
   //L0
   fmul v11.4s,v18.4s,v24.s[0]   //k00
   fmla v11.4s,v15.4s,v24.s[1]   //k01
   fmla v11.4s,v19.4s,v24.s[2]   //k02
   st1 {v11.4s},[x6],#16
  
   ins v17.s[3],v15.s[3]

mid_less_16:

   sub x8,x2,x7
   cmp x8,mid_row_done
   lsr x8,x8,2
   add x7,x7,x8,LSL #2
   cbz x8, mid_less_4

mid_loop_4_start:
 
   ld1 {v0.4s},[x4]
   ld1 {v4.4s},[x5]
   
  ld1 {v12.4s},[x0],#16

  cmp x8,2
  bge do_load4

  ins v13.s[0],v31.s[0]
  cmp x7,x2
  bge skip_load4

do_load4:
  ld1r {v13.4s},[x0]

skip_load4:
  
  ext v28.16b,v17.16b,v12.16b,12  // last_3 , a00, a01, a02
                          //v12: a00, a01, a02 ,a03 
  ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04   
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
  
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   st1 {v8.4s},[x6],#16

   ins v17.s[3],v12.s[3]
   
   subs x8,x8,1   
   b.ne mid_loop_4_start
 
mid_less_4:
   sub x8,x2,x7
   cmp x8,1
   blt  mid_row_start
   
mid_left_1_2_3: 
  
   dup v12.4s,v31.s[0]
   dup v13.4s,v31.s[0]
   dup v0.4s,v31.s[0]
   dup v4.4s,v31.s[0]
   
   
   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
      
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
   ldr s28,[x5]
   ins v4.s[0],v28.s[0]
   
   
   cmp  x8,2
   blt mid_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
   
   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]
   ldr s28,[x5, #4]
   ins v4.s[1],v28.s[0]
   
   cmp  x8,3
   blt mid_left_load_done
   
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
   ldr s28,[x4,#8]
   ins v0.s[2],v28.s[0]
   ldr s28,[x5, #8]
   ins v4.s[2],v28.s[0]

mid_left_load_done:         

   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v0.4s,v29.4s,v26.s[2]  //k22
 
  
   //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12

   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   
   //save result:1, 2 or 3
   ins v28.s[0],v0.s[0]
   str  s28,[x4],#4
   
   ins v28.s[0],v4.s[0]
   str  s28,[x5],#4
   
   ins v28.s[0],v8.s[0]
   str s28,[x6],#4
   
   cmp x8,2
   blt mid_row_start
   
   ins v28.s[0],v0.s[1]
   str s28,[x4],#4

   ins v28.s[0],v4.s[1]
   str s28,[x5],#4
   
   ins v28.s[0],v8.s[1]
   str s28,[x6],#4
   
   cmp x8,3
   blt mid_row_start

   ins v28.s[0],v0.s[2]
   str s28,[x4],#4
  
   ins v28.s[0],v4.s[2]
   str s28,[x5]
   
   ins v28.s[0],v8.s[2]
   str s28,[x6]
   
   b mid_row_start

mid_row_done:

 //Last row  

last_row_start:
   
   lsr  x8,x2,4
   lsl  x7,x8,4
   
   dup v17.4s,v31.s[0]
   
   add x5,x4,x9 //L-1

   cbz x8,last_less_16
   sub x8,x8,1
   cbz x8,last_last_16_block
   
last_loop_start:
   
   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4]  //L-2
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x5]  //L-1
  
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   ld1r {v16.4s},[x0]
  
   //block 0: v12
   
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
  
   //block 1: v13
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v1.4s,v18.4s,v26.s[0]  //k20, 
  fmla v1.4s,v13.4s,v26.s[1]  //k21,
  fmla v1.4s,v19.4s,v26.s[2]  //k22
  st1 {v1.4s},[x4],#16
  
  
  //L-1   
   fmla v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
   
   //block 2: v14
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v2.4s,v28.4s,v26.s[0]  //k20, 
  fmla v2.4s,v14.4s,v26.s[1]  //k21,
  fmla v2.4s,v29.4s,v26.s[2]  //k22
  st1 {v2.4s},[x4],#16
  
  
  //L-1   
   fmla v6.4s,v28.4s,v25.s[0]  //k10, 
   fmla v6.4s,v14.4s,v25.s[1]  //k11,
   fmla v6.4s,v29.4s,v25.s[2]  //k12
   st1 {v6.4s},[x5],#16
  
   //block 3: v15
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v3.4s,v18.4s,v26.s[0]  //k20, 
  fmla v3.4s,v15.4s,v26.s[1]  //k21,
  fmla v3.4s,v19.4s,v26.s[2]  //k22
  st1 {v3.4s},[x4],#16
  
  
  //L-1   
   fmla v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
  
   ins v17.s[3],v15.s[3]
   
   //next loop
   subs x8,x8,1
   b.ne last_loop_start


last_last_16_block:

   ld1 {v0.4s,v1.4s,v2.4s,v3.4s},[x4]  //L-2
   ld1 {v4.4s,v5.4s,v6.4s,v7.4s},[x5]  //L-1
  
   ld1 {v12.4s,v13.4s,v14.4s,v15.4s},[x0],#64
   dup v16.4s,v31.s[0]
   cmp x7,x2
   bge skip_load5
   ld1r {v16.4s},[x0]

 skip_load5:
 
   //block 0: v12
   
   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16
  
   //block 1: v13
   ext v18.16b,v12.16b,v13.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v13.16b,v14.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v1.4s,v18.4s,v26.s[0]  //k20, 
  fmla v1.4s,v13.4s,v26.s[1]  //k21,
  fmla v1.4s,v19.4s,v26.s[2]  //k22
  st1 {v1.4s},[x4],#16
  
  
  //L-1   
   fmla v5.4s,v18.4s,v25.s[0]  //k10, 
   fmla v5.4s,v13.4s,v25.s[1]  //k11,
   fmla v5.4s,v19.4s,v25.s[2]  //k12
   st1 {v5.4s},[x5],#16
   
   //block 2: v14
   ext v28.16b,v13.16b,v14.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v14.16b,v15.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v2.4s,v28.4s,v26.s[0]  //k20, 
  fmla v2.4s,v14.4s,v26.s[1]  //k21,
  fmla v2.4s,v29.4s,v26.s[2]  //k22
  st1 {v2.4s},[x4],#16
  
  
  //L-1   
   fmla v6.4s,v28.4s,v25.s[0]  //k10, 
   fmla v6.4s,v14.4s,v25.s[1]  //k11,
   fmla v6.4s,v29.4s,v25.s[2]  //k12
   st1 {v6.4s},[x5],#16
  
   //block 3: v15
   ext v18.16b,v14.16b,v15.16b,12  //last_3 , a00, a01, a02
   ext v19.16b,v15.16b,v16.16b,4   //a01, a02, a03, a04
  
  //L-2 
  fmla v3.4s,v18.4s,v26.s[0]  //k20, 
  fmla v3.4s,v15.4s,v26.s[1]  //k21,
  fmla v3.4s,v19.4s,v26.s[2]  //k22
  st1 {v3.4s},[x4],#16
  
  
  //L-1   
   fmla v7.4s,v18.4s,v25.s[0]  //k10, 
   fmla v7.4s,v15.4s,v25.s[1]  //k11,
   fmla v7.4s,v19.4s,v25.s[2]  //k12
   st1 {v7.4s},[x5],#16
  
   ins v17.s[3],v15.s[3]

last_less_16:

   sub x8,x2,x7
   lsr x8,x8,2
   cbz x8, last_less_4
   add x7,x7,x8,LSL #2
 
 last_loop_4_start:
 
   ld1 {v0.4s},[x4]
   ld1 {v4.4s},[x5]
   
   ld1 {v12.4s},[x0],#16

   cmp x8,2
   bge do_load6
   ins v13.s[0],v31.s[0]
   cmp x7,x2
   bge skip_load6

do_load6:
   ld1r {v13.4s},[x0]

skip_load6:
  
  ext v28.16b,v17.16b,v12.16b,12  // last_3 , a00, a01, a02
                          //v12: a00, a01, a02 ,a03 
  ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04   
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   st1 {v4.4s},[x5],#16

  
   ins v17.s[3],v12.s[3]
 
   subs x8,x8,1   
   b.ne last_loop_4_start
    
last_less_4:
   sub x8,x2,x7   
   cmp x8,1
   blt last_row_done

last_1_2_3:   
  
   dup v12.4s,v31.s[0]
   dup v13.4s,v31.s[0]
   dup v0.4s,v31.s[0]
   dup v4.4s,v31.s[0]
   

   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
   ldr s28,[x5]
   ins v4.s[0],v28.s[0]
  
   sub x7,x8,1
   cbz x7, last_left_load_done
    
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]

   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]
   ldr s28,[x5,#4]
   ins v4.s[1],v28.s[0]
   
   
   sub x7,x8,2
   cbz x7, last_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]

   ldr s28,[x4,#8]
   ins v0.s[2],v28.s[0]
   ldr s28,[x5,#8]
   ins v4.s[2],v28.s[0]

last_left_load_done:         

   ext v28.16b,v17.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v0.4s,v29.4s,v26.s[2]  //k22
  
   //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   
   //save result: 1 2 or 3
   ins v28.s[0],v0.s[0]
   str  s28,[x4],#4

   ins v28.s[0],v4.s[0]
   str  s28,[x5],#4
 
   cmp x8,2
   blt last_row_done
 
   ins v28.s[0],v0.s[1]
   str s28,[x4],#4
  
   ins v28.s[0],v4.s[1]
   str s28,[x5],#4
   
   cmp x8,3
   blt last_row_done

   ins v28.s[0],v0.s[2]
   str s28,[x4]
   
   ins v28.s[0],v4.s[2]
   str s28,[x5]
   
   
last_row_done:
   ret










