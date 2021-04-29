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
 * Author: haitao@openailab.com
 */


//x0: input
//x1: h
//x2: w
//x3: kernel
//x4: output //L-2
//x5 : bias
//x10: L-1 output
//x6: L0 output
//x7: processed item
//x8: counter
//x9: x2*4 

//v0-v3: L-2  
//v4-v7: L-1  
//v8-v11: L0  
//v12-v15: input group
//v24-v26: kernel
//v27 --- saved previous vector
// v28,v29 --- shifted 

//v30 : bias
//v31: relu vector

//v18: activation vection
//x18: activation variable

#define CONV_RELU_FUSE

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s1p1_a72
#endif

.text
.align 5
.global KERNEL_NAME
.hidden KERNEL_NAME
.type KERNEL_NAME, %function


KERNEL_NAME:

   mov x18,x6
   scvtf s18,w18
   dup v18.4s,v18.s[0]
   
   //Load Kernel
   ld1 {v24.4s,v25.4s,v26.4s}, [x3]

   ext  v26.16b,v25.16b,v26.16b,8
   ext  v25.16b,v24.16b,v25.16b,12

   sub sp,sp,#0x40
   stp d8,d9,[sp]
   stp d10,d11,[sp,0x10]
   stp d12,d13,[sp,0x20]
   stp d14,d15,[sp,0x30]

   lsl x9,x2,2
   fmov s31,wzr
   dup  v31.4s,v31.s[0]

   cbz  x5 ,non_biases
   //get the bias
   ldr  s30, [x5]
   dup  v30.4s,v30.s[0] 

   b  first_row_start
non_biases:
   fmov s30, wzr
   dup v30.4s,v30.s[0]
first_row_start:
   sub  x1,x1,1  
   sub  x7,x2,1  //save last item in row  
   lsr  x8,x7,2
   lsl  x7,x8,2

   ins  v27.s[3],v31.s[0]   //pre_vector for input

   mov x10,x4      //L-1
   add x6,x10,x9   //L-0


   cbz  x8,first_last_4

   //output
   
   
first_row_loop:
   //load 4 float input
   ld1 {v12.4s},[x0],#16
   ld1r {v13.4s},[x0]
   
   ext v28.16b,v27.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04
   
   ins v27.s[3],v12.s[3]  //save prev vector
   
   //L-1: k1 xinput
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
  
   st1 {v4.4s},[x10],#16
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   
   st1 {v8.4s},[x6],#16
   
   //next loop
   subs x8,x8,1
   b.ne first_row_loop

first_last_4:
   //left ones: 1-4
   sub x8,x2,x7
   cmp x8,4
   blt first_less_4

   //4  nodes
   ld1 {v12.4s},[x0],#16
   ins v13.s[0],v31.s[0]
   
   ext v28.16b,v27.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         
   
   //L-1: k1 xinput
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   
   st1 {v4.4s},[x10],#16
   
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   
   st1 {v8.4s},[x6],#16
  
   b first_row_done
   
first_less_4:
   cmp x8,1
   bge first_1_2_3
   b   first_row_done

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

   ext v28.16b,v27.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-1   
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v8.4s,v12.4s,v24.s[1]   //k01
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   fmla v8.4s,v29.4s,v24.s[2]   //k02
   
   //save result: 2 or 3
   ins v28.s[0],v4.s[0]
   str  s28,[x10],#4
   
   ins v28.s[0],v8.s[0]
   str s28,[x6],#4
   
   cmp x8, 2
   blt  first_row_done
   
  ins v28.s[0],v4.s[1]
  str s28,[x10],#4
   
   
   ins v28.s[0],v8.s[1]
   str s28,[x6],#4
   
   cmp x8,3
   blt first_row_done
   
   ins v28.s[0],v4.s[2]
   str s28,[x10]
   
   ins v28.s[0],v8.s[2]
   str s28,[x6]

first_row_done:
   
mid_row_start:

   sub x1,x1,1
   cbz x1, last_row_start

   sub  x7,x2,1  //save one 
   lsr  x8,x7,2
   lsl  x7,x8,2
   
   add x10,x4,x9    //L-1
   add x6,x10,x9   //L0
   dup v27.4s,v31.s[0]
     
   cbz x8,mid_last_4
  
mid_loop_start:
   
   ld1 {v0.4s},[x4]
   ld1 {v4.4s},[x10]
  //ld1 {v8.4s},[x6],#16  //L0 is always zero
  
  ld1 {v12.4s},[x0],#16
  ld1r {v13.4s},[x0]
 
  ext v28.16b,v27.16b,v12.16b,12  // last_3 , a00, a01, a02
                          //v12: a00, a01, a02 ,a03 
  ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04   
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmul v8.4s,v28.4s,v24.s[0]   //k00
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v8.4s,v12.4s,v24.s[1]   //k01
  fmla v0.4s,v29.4s,v26.s[2]  //k22
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   fmla v8.4s,v29.4s,v24.s[2]   //k02
//add bias
  fadd v0.4s,v0.4s,v30.4s
#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v0.4s,v0.4s,v31.4s
    beq 100f
    fmin v0.4s,v0.4s,v18.4s
100:

#endif
  st1 {v0.4s},[x4],#16
  
  //L-1   
   st1 {v4.4s},[x10],#16
  
   
   //L0
   st1 {v8.4s},[x6],#16
  
   ins v27.s[3],v12.s[3]
   
   //next loop
   subs x8,x8,1
   b.ne mid_loop_start

mid_last_4:
   sub x8,x2,x7
   cmp x8,4
   blt mid_less_4
   
   ld1 {v0.4s},[x4]
   ld1 {v4.4s},[x10]
   
  ld1 {v12.4s},[x0],#16
  ins v13.s[0],v31.s[0]
  
  ext v28.16b,v27.16b,v12.16b,12  // last_3 , a00, a01, a02
                          //v12: a00, a01, a02 ,a03 
  ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04   
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmul v8.4s,v28.4s,v24.s[0]   //k00
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v8.4s,v12.4s,v24.s[1]   //k01
  fmla v0.4s,v29.4s,v26.s[2]  //k22
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   fmla v8.4s,v29.4s,v24.s[2]   //k02
//add bias
  fadd v0.4s,v0.4s,v30.4s
#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v0.4s,v0.4s,v31.4s
    beq 100f
    fmin v0.4s,v0.4s,v18.4s
100:

#endif
  st1 {v0.4s},[x4],#16
  
  
  //L-1   
   st1 {v4.4s},[x10],#16
  
   //L0
   st1 {v8.4s},[x6],#16
   
   b mid_row_start
 
mid_less_4:
   cmp x8,1
   blt mid_row_start
   
mid_left_1_2_3: 
  
   dup v12.4s,v31.s[0]
   dup v13.4s,v31.s[0]
   dup v0.4s,v31.s[0]
   dup v4.4s,v31.s[0]
   
   
   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
      
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
   ldr s28,[x10]
   ins v4.s[0],v28.s[0]
   
   
   cmp  x8,2
   blt mid_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
   
   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]
   ldr s28,[x10, #4]
   ins v4.s[1],v28.s[0]
   
   cmp  x8,3
   blt mid_left_load_done
   
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
   ldr s28,[x4,#8]
   ins v0.s[2],v28.s[0]
   ldr s28,[x10, #8]
   ins v4.s[2],v28.s[0]

mid_left_load_done:         

   ext v28.16b,v27.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v8.4s,v12.4s,v24.s[1]   //k01

   fmla v0.4s,v29.4s,v26.s[2]  //k22
   fmla v4.4s,v29.4s,v25.s[2]  //k12
   fmla v8.4s,v29.4s,v24.s[2]   //k02

//add bias 
   fadd v0.4s,v0.4s,v30.4s
   //save result:1, 2 or 3
   ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str  s28,[x4],#4
   
   ins v28.s[0],v4.s[0]
   str  s28,[x10],#4
   
   ins v28.s[0],v8.s[0]
   str s28,[x6],#4
   
   cmp x8,2
   blt mid_row_start
   
   ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x4],#4

   ins v28.s[0],v4.s[1]
   str s28,[x10],#4
   
   ins v28.s[0],v8.s[1]
   str s28,[x6],#4
   
   cmp x8,3
   blt mid_row_start

   ins v28.s[0],v0.s[2]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x4],#4
  
   ins v28.s[0],v4.s[2]
   str s28,[x10]
   
   ins v28.s[0],v8.s[2]
   str s28,[x6]
   
   b mid_row_start
   

last_row_start:

   
   sub  x7,x2,1
   lsr  x8,x7,2
   lsl  x7,x8,2
   
   dup v27.4s,v31.s[0]
   
   add x10,x4,x9 //L-1

   cbz x8,last_last_4
   
last_loop_start:
   
  ld1 {v0.4s},[x4]
  ld1 {v4.4s},[x10]
  
  ld1 {v12.4s},[x0],#16
  ld1 {v13.4s},[x0]
  
  ext v28.16b,v27.16b,v12.16b,12  // last_3 , a00, a01, a02
                           //v12: a00, a01, a02 ,a03 
  ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04   
  
  //L-2 
  fmla v0.4s,v28.4s,v26.s[0]  //k20, 
  fmla v4.4s,v28.4s,v25.s[0]  //k10, 
  fmla v0.4s,v12.4s,v26.s[1]  //k21,
  fmla v4.4s,v12.4s,v25.s[1]  //k11,
  fmla v0.4s,v29.4s,v26.s[2]  //k22
  fmla v4.4s,v29.4s,v25.s[2]  //k12
//add bias
  fadd v0.4s,v0.4s,v30.4s

#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v0.4s,v0.4s,v31.4s
    beq 100f
    fmin v0.4s,v0.4s,v18.4s
100:

#endif
  st1 {v0.4s},[x4],#16
  
  //L-1   
//add bias
   fadd v4.4s,v4.4s,v30.4s
#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v4.4s,v4.4s,v31.4s
    beq 100f
    fmin v4.4s,v4.4s,v18.4s
100:

#endif
   st1 {v4.4s},[x10],#16
  
   ins v27.s[3],v12.s[3]
   
   //next loop
   subs x8,x8,1
   b.ne last_loop_start

last_last_4:

   sub x8,x2,x7
   cmp x8,4
   blt last_less_4
  
   ld1 {v12.4s},[x0],#16
   dup v13.4s,v31.s[0]
  
   ext v28.16b,v27.16b,v12.16b,12  // last_3 , a00, a01, a02
                           //v12: a00, a01, a02 ,a03 
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04      

   ld1 {v0.4s},[x4]
   ld1 {v4.4s},[x10]
   
   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v0.4s,v29.4s,v26.s[2]  //k22
//add bias
   fadd v0.4s,v0.4s,v30.4s   

#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v0.4s,v0.4s,v31.4s
    beq 100f
    fmin v0.4s,v0.4s,v18.4s
100:

#endif
   st1 {v0.4s},[x4],#16
  
  
   //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12
//add bias
   fadd v4.4s,v4.4s,v30.4s
#ifdef CONV_RELU_FUSE
    cmp w18,0
    blt 100f
    fmax v4.4s,v4.4s,v31.4s
    beq 100f
    fmin v4.4s,v4.4s,v18.4s
100:

#endif
   st1 {v4.4s},[x10],#16
  
   ins v27.s[3],v12.s[3]
 
   b last_row_done
    
last_less_4:
      
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
   ldr s28,[x10]
   ins v4.s[0],v28.s[0]
  
   sub x7,x8,1
   cbz x7, last_left_load_done
    
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]

   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]
   ldr s28,[x10,#4]
   ins v4.s[1],v28.s[0]
   
   
   sub x7,x8,2
   cbz x7, last_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]

   ldr s28,[x4,#8]
   ins v0.s[2],v28.s[0]
   ldr s28,[x10,#8]
   ins v4.s[2],v28.s[0]

last_left_load_done:         

   ext v28.16b,v27.16b,v12.16b,12  //last_3 , a00, a01, a02
   ext v29.16b,v12.16b,v13.16b,4   //a01, a02, a03, a04         

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v12.4s,v26.s[1]  //k21,
   fmla v0.4s,v29.4s,v26.s[2]  //k22
  
   //L-1   
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v12.4s,v25.s[1]  //k11,
   fmla v4.4s,v29.4s,v25.s[2]  //k12

//add bias
   fadd v0.4s,v0.4s,v30.4s   
   //save result: 1 2 or 3
   ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str  s28,[x4],#4

//add bias
   fadd v4.4s,v4.4s,v30.4s

   ins v28.s[0],v4.s[0]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str  s28,[x10],#4
 
   cmp x8,2
   blt last_row_done
 
   ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x4],#4
  
   ins v28.s[0],v4.s[1]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x10],#4
   
   cmp x8,3
   blt last_row_done

   ins v28.s[0],v0.s[2]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x4]
   
   ins v28.s[0],v4.s[2]
#ifdef CONV_RELU_FUSE
   cmp w18,0
   blt 100f
   fmax s28,s28,s31
   beq 100f
   fmin s28,s28,s18
100:

#endif
   str s28,[x10]
   
   
last_row_done:
    ldp d8,d9,[sp]
    ldp d10,d11,[sp,0x10]
    ldp d12,d13,[sp,0x20]
    ldp d14,d15,[sp,0x30]
 
    add sp,sp,#0x40


   ret









