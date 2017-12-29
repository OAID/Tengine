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
//x9: output width

//v0-v3: L-2  
//v4-v7: L-1  
//v8-v11: L0  
//v12-v15/v16-v20: input two group
//v24-v26: kernel
//v27 --- saved previous vector
// v28,v29 --- shifted 

#ifndef KERNEL_NAME
#define KERNEL_NAME dw_k3s2p1
#endif

.text
.align 5
.global KERNEL_NAME
.type KERNEL_NAME, %function


KERNEL_NAME:
   //Load Kernel
   ld1 {v24.4s,v25.4s,v26.4s}, [x3]
   ext  v26.16b,v25.16b,v26.16b,8
   ext  v25.16b,v24.16b,v25.16b,12

   sub x9,x2,1
   lsr x9,x9,1
   add x9,x9,1
   lsl x9,x9,2
   fmov s31,wzr
   dup  v31.4s,v31.s[0]

//first row

first_row_start:
   sub  x1,x1,1
     
   lsr  x8,x2,3    //x8 loop counter
   lsl  x7,x8,3    //x7 processed number

   ins  v27.s[3],v31.s[0]   //pre_vector for input

   mov x5,x4      //L-1  //L1 ONLY
   cbz  x8,first_less_8
   
first_loop_start:
   //load 4 float input
   ld1 {v12.4s,v13.4s},[x0],#32    //a00,a01,a02,a03,a04,a05,a06,a07
   
   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05
   
   /*
     v28:   last_3, a01, a03, a05
     v29    a00     a02,  a04, a06
     v30    a01     a03,  a05, a07
   */  
   
   //L-1: k1 xinput
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v29.4s,v25.s[1]  //k11,
   fmla v4.4s,v30.4s,v25.s[2]  //k12

   ins v27.s[3],v13.s[3]  //save prev vector

    //save data, four are valid
#ifdef CONV_RELU_FUSE
    fmax v4.4s,v4.4s,v31.4s
#endif
    st1 {v4.4s},[x5],#16
   
    //next loop
    subs x8,x8,1
    b.ne first_loop_start

first_less_8:
   
    sub x8,x2,x7
    cmp  x8,1
    blt first_row_done

first_1_7:
    dup v13.4s,v31.s[0]

    cmp x8,4
    blt  first_1_2_3
    
    ld1 {v12.4s},[x0],#16

    uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
    uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
    ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

    //L-1   
    fmul v4.4s,v28.4s,v25.s[0]  //k10, 
    fmla v4.4s,v29.4s,v25.s[1]  //k11,
    fmla v4.4s,v30.4s,v25.s[2]  //k12

    ins v28.s[0],v4.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x5],#4

    ins v28.s[0],v4.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x5],#4

    sub x8,x8,4
    cbz x8,first_row_done

    ins v27.s[3],v12.s[3]
     
first_1_2_3:
    dup v12.4s,v31.s[0]

    //1-3 items
    ldr s28,[x0],#4
    ins v12.s[0],v28.s[0]

    cmp x8,2
    blt first_left_load_done

    ldr s28,[x0],#4
    ins v12.s[1],v28.s[0]

    cmp x8,3
    blt first_left_load_done

   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
first_left_load_done:         

   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05
   
   //L-1   
   fmul v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v29.4s,v25.s[1]  //k11,
   fmla v4.4s,v30.4s,v25.s[2]  //k12

first_left_save_1_3:  
   
   ins v28.s[0],v4.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str  s28,[x5],#4

   cmp x8,3
   blt first_row_done
   
   ins v28.s[0],v4.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str s28,[x5],#4

first_row_done:


odd_row_start:
   sub x1,x1,1
   cbz x1, last_row_is_odd

   lsr  x8,x2,3
   lsl  x7,x8,3
   
   dup v27.4s,v31.s[0]
                   //x4: L-2
   add x6,x4,x9   //L0
     
   cbz x8,odd_less_8
  
odd_loop_start:

   ld1 {v0.4s}, [x4]   //L-2
   ld1 {v12.4s,v13.4s},[x0],#32
   
   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05
 
  
  //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v29.4s,v26.s[1]  //k21,
   fmla v0.4s,v30.4s,v26.s[2]  //k22
#ifdef CONV_RELU_FUSE
   fmax v0.4s,v0.4s,v31.4s
#endif
     
   //L0 is always zero
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v29.4s,v24.s[1]   //k01
   fmla v8.4s,v30.4s,v24.s[2]   //k02
#ifdef CONV_RELU_FUSE
   fmax v8.4s,v8.4s,v31.4s
#endif
   
   st1 {v0.4s}, [x4],#16
   st1 {v8.4s}, [x6],#16
      
   ins v27.s[3],v13.s[3]
   
   //next loop
   subs x8,x8,1
   b.ne odd_loop_start

odd_less_8:
   sub x8,x2,x7
   cmp x8,1
   blt odd_row_done

odd_1_7:
    dup v13.4s,v31.s[0]
    cmp x8,4
    blt  odd_1_2_3

    ld1 {v12.4s},[x0],#16

    ldr s28,[x4]
    ins v0.s[0],v28.s[0]
   
    ldr s28,[x4,#4]
    ins v0.s[1],v28.s[0]

    uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
    uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
    ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05
 
    //L-2 
    fmla v0.4s,v28.4s,v26.s[0]  //k20, 
    fmla v0.4s,v29.4s,v26.s[1]  //k21,
    fmla v0.4s,v30.4s,v26.s[2]  //k22
     
    //L0 is always zero
    fmul v8.4s,v28.4s,v24.s[0]   //k00
    fmla v8.4s,v29.4s,v24.s[1]   //k01
    fmla v8.4s,v30.4s,v24.s[2]   //k02  

    ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x4],#4

    ins v28.s[0],v8.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x6],#4
  
    ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str s28,[x4],#4

    ins v28.s[0],v8.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x6],#4

    sub x8,x8,4
    cbz x8, odd_row_done

    ins v27.s[3],v12.s[3]

odd_1_2_3:

   dup v12.4s,v31.s[0]

   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
   
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
  
   cmp  x8,2
   blt odd_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
     
   cmp  x8,3
   blt odd_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]

odd_left_load_done:         

   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05
 
  
   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v29.4s,v26.s[1]  //k21,
   fmla v0.4s,v30.4s,v26.s[2]  //k22
     
   //L0
   fmul v8.4s,v28.4s,v24.s[0]   //k00
   fmla v8.4s,v29.4s,v24.s[1]   //k01
   fmla v8.4s,v30.4s,v24.s[2]   //k02
   
   //save result:1 or 2
   ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str  s28,[x4],#4

   ins v28.s[0],v8.s[0]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str  s28,[x6],#4

   cmp x8,3
   blt odd_row_done
   
   ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str s28,[x4],#4

   ins v28.s[0],v8.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
   str  s28,[x6],#4

odd_row_done:   

even_row_start:

   lsr  x8,x2,3
   lsl  x7,x8,3

   ins  v27.s[3],v31.s[0]   //pre_vector for input

   mov x5,x4       //L-1  //L1 ONLY
   cbz  x8,even_less_8
   
even_loop_start:
   //load 4 float input
   ld1 {v12.4s,v13.4s},[x0],#32      
   ld1 {v4.4s},[x5]
   
   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

    //L-1: k1 xinput
   fmla v4.4s,v28.4s,v25.s[0]  //k10, 
   fmla v4.4s,v29.4s,v25.s[1]  //k11,
   fmla v4.4s,v30.4s,v25.s[2]  //k12
#ifdef CONV_RELU_FUSE
   fmax v4.4s,v4.4s,v31.4s
#endif

   ins v27.s[3],v13.s[3]  //save prev vector

   st1 {v4.4s},[x5],#16
   
   //next loop
   subs x8,x8,1
   b.ne even_loop_start

even_less_8:
   
   sub x8,x2,x7
   cmp  x8,1
   blt even_row_done

even_1_7:
    dup v13.4s,v31.s[0]
    
    cmp x8,4
    blt  even_1_2_3

    ld1 {v12.4s},[x0],#16
    ldr s28,[x5]
    ins v4.s[0],v28.s[0]
    ldr s28,[x5,#4]
    ins v4.s[1],v28.s[0]
   
    uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
    uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
    ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

    //L-1: k1 xinput
    fmla v4.4s,v28.4s,v25.s[0]  //k10, 
    fmla v4.4s,v29.4s,v25.s[1]  //k11,
    fmla v4.4s,v30.4s,v25.s[2]  //k12

    ins v28.s[0],v4.s[0]
    str  s28,[x5],#4
    
    ins v28.s[0],v4.s[1]
#ifdef CONV_RELU_FUSE
    fmax s28,s28,s31
#endif
    str  s28,[x5],#4

    sub x8,x8,4
    cbz x8, even_row_done

    ins v27.s[3],v12.s[3]  //save prev vector

even_1_2_3:   
   dup v12.4s,v31.s[0]
   
   //1, 2 or 3 items
   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]

   ldr s28,[x5]
   ins v4.s[0],v28.s[0]
   
   sub x7,x8,1
   cbz x7, even_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
  
   sub x7,x8,2
   cbz x7, even_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
   ldr s28,[x5,#4]
   ins v4.s[1],v28.s[0]

even_left_load_done:         

    uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
    uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
    ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

    //L-1: k1 xinput
    fmla v4.4s,v28.4s,v25.s[0]  //k10, 
    fmla v4.4s,v29.4s,v25.s[1]  //k11,
    fmla v4.4s,v30.4s,v25.s[2]  //k12

      
   //save result: 1 or 2
   ins v28.s[0],v4.s[0]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str  s28,[x5],#4

   cmp x8,3
   blt even_row_done
   
   ins v28.s[0],v4.s[1]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str s28,[x5],#4

even_row_done:
   sub  x1,x1,1  
   cbz x1, all_row_done 
   b odd_row_start


// Last Row: even or odd

last_row_is_odd:
  
   lsr  x8,x2,3
   lsl  x7,x8,3
   
   dup v27.4s,v31.s[0]
   cbz x8,last_odd_less_8
   
last_odd_loop_start:

   ld1 {v0.4s},[x4]   //L-2
   ld1 {v12.4s,v13.4s},[x0],#32
  
   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v29.4s,v26.s[1]  //k21,
   fmla v0.4s,v30.4s,v26.s[2]  //k22

#ifdef CONV_RELU_FUSE
   fmax v0.4s,v0.4s,v31.4s
#endif
   st1 {v0.4s},[x4],#16
      
   ins v27.s[3],v13.s[3]
   
   //next loop
   subs x8,x8,1
   b.ne last_odd_loop_start

last_odd_less_8:
   sub x8,x2,x7
   cmp x8,1
   blt last_odd_row_done
   cmp x8,4
   blt last_odd_1_2_3

   ld1 {v12.4s},[x0],#16
   dup v13.4s,v31.s[0]

    //L-2
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]
   
   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v29.4s,v26.s[1]  //k21,
   fmla v0.4s,v30.4s,v26.s[2]  //k22

   ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str  s28,[x4],#4

   ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str s28,[x4],#4

   sub x8,x8,4
   cbz x8,last_odd_row_done
   
   ins v27.s[3],v12.s[3]
  
last_odd_1_2_3:

   dup v12.4s,v31.s[0]
       
   ldr s28,[x0],#4
   ins v12.s[0],v28.s[0]
   
   ldr s28,[x4]
   ins v0.s[0],v28.s[0]
  
   cmp  x8,2
   blt last_odd_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[1],v28.s[0]
     
   cmp  x8,3
   blt last_odd_left_load_done
   
   ldr s28,[x0],#4
   ins v12.s[2],v28.s[0]
   
   ldr s28,[x4,#4]
   ins v0.s[1],v28.s[0]

last_odd_left_load_done:         

   uzp1 v29.4s,v12.4s,v13.4s  //a00,a02,a04,a06
   uzp2 v30.4s,v12.4s,v13.4s  //a01,a03,a05,a07
   ext v28.16b,v27.16b,v30.16b,12  //last_3 , a01, a03,a05

   //L-2 
   fmla v0.4s,v28.4s,v26.s[0]  //k20, 
   fmla v0.4s,v29.4s,v26.s[1]  //k21,
   fmla v0.4s,v30.4s,v26.s[2]  //k22

     
   //save result:1 or 2
   ins v28.s[0],v0.s[0]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str  s28,[x4],#4


   cmp x8,3
   blt last_odd_row_done
   
   ins v28.s[0],v0.s[1]
#ifdef CONV_RELU_FUSE
   fmax s28,s28,s31
#endif
   str s28,[x4],#4


last_odd_row_done:   
all_row_done:
   ret




