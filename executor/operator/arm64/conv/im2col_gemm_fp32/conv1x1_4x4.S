// register definition
// x0        bias start address
// x1        input start address
// x2        kernel start address
// x3        output start address
// x4        in_hw
// x5        c_in
// x6        fuse_relu

// x9        cin/4
// x11       resi_cin

// x12       inp+ hw
// x13       inp+ 2*hw
// x14       inp+ 3*hw



// v0 input[j][0,1,2,3]
// v1 input[j+1][0,1,2,3]
// v2 input[j+2][0,1,2,3]
// v3 input[j+3][0,1,2,3]
// v4 ker[i][0,1,2,3]
// v5 ker[i+1][0,1,2,3]
// v6 ker[i+2][0,1,2,3]
// v7 ker[i+3][0,1,2,3]
// v16 output[i][0,1,2,3]
// v17 output[i+1][0,1,2,3]
// v18 output[i+2][0,1,2,3]
// v19 output[i+3][0,1,2,3]
// v8 ~ v15 not used
// v20 ~ v31 not used

	.section .text,"ax"
	.align 5

	.type conv1x1_4x4 STT_FUNC
	.global conv1x1_4x4

conv1x1_4x4:
    prfm	pldl1keep, [x1]
    lsl x4,x4,2                     // hw * sizeof(float)
	cbz	x0, none_biases             // cbz R0, branch   -- if R0==0, go to branch
    ld4r    {v16.4s,v17.4s,v18.4s,v19.4s}, [x0]
    b convolution_start

none_biases:
	movi	d16, 0x0
	movi	d17, 0x0
	movi	d18, 0x0
	movi	d19, 0x0

convolution_start:
    cmp x5,0x4    
    b.lt loop4_end
  

    add x12, x1, x4                     // inp2 = inp + hw *sizeof(float)           
    lsr	x9, x5, 0x2                          // X9 = c_in / 4
    add x13, x12, x4                    // inp3 = inp + hw * 2 *sizeof(float)
    add x14, x13, x4                    // inp4 = inp + hw * 3 *sizeof(float)
    mov    x7,#12
    mul    x7,x7,x4
   
loop4:  
    ldr	q0, [x1]	                         // inp0 = inp 
    ldr	q1, [x12]	                         // inp1 = inp0 + hw
    ldp q4,q5, [x2] 
    ldr	q2, [x13]	                         // inp2 = inp1 + hw
    ldr	q3, [x14]	                         // inp3 = inp2 + hw
    ldp q6,q7,[x2,0x20]
    subs	x9, x9, 0x1

    fmla	v16.4s, v0.4s,  v4.s[0]	
    fmla	v17.4s, v1.4s,  v5.s[1]	
    prfm    pldl1keep, [x1, x7]
    fmla	v18.4s, v2.4s,  v6.s[2]	
	fmla	v19.4s, v3.4s,  v7.s[3]	
    add x1, x1,  x4,LSL 2                      //    inp0 += 4*hw;

	fmla	v16.4s, v1.4s,  v4.s[1]
    prfm    pldl1keep, [x12, x7]    
	fmla	v17.4s, v2.4s,  v5.s[2]	
     add x12,x12, x4,LSL 2                       //    inp1 += 4*hw;
	fmla	v18.4s, v3.4s,  v6.s[3]	
	fmla	v19.4s, v0.4s,  v7.s[0]	

	prfm	pldl1keep, [x2, 0x140]
    add x2,x2,0x40
	fmla	v16.4s, v2.4s,  v4.s[2]	
	fmla	v17.4s, v3.4s,  v5.s[3]	
    prfm    pldl1keep, [x13, x7]    
    fmla	v18.4s, v0.4s,  v6.s[0]	
    fmla	v19.4s, v1.4s,  v7.s[1]	
    add x13,x13, x4,LSL 2                       //    inp2 += 4*hw;
	fmla	v16.4s, v3.4s,  v4.s[3]	
    prfm    pldl1keep, [x14, x7]    
	fmla	v17.4s, v0.4s,  v5.s[0]	
    add x14,x14, x4,LSL 2                       //    inp3 += 4*hw;    
	fmla	v18.4s, v1.4s,  v6.s[1]	
	fmla	v19.4s, v2.4s,  v7.s[2]	                                           
	b.ne	loop4

loop4_end:
    and x11,x5,0x3                         // x11= c_in&~3, if resi, go to 
    cbz x11, fuse_relu

loop1:
    ldr	q0, [x1]   // inp0   
    ldr q4, [x2]
    subs x11,x11,0x1
    fmla	v16.4s, v0.4s,  v4.s[0]	
    fmla	v17.4s, v0.4s,  v4.s[1]
    add x1, x1, x4                              //    inp0 += hw;
    fmla	v18.4s, v0.4s,  v4.s[2]	
    add	x2, x2, 0x10
    fmla	v19.4s, v0.4s,  v4.s[3]	
    b.ne loop1

fuse_relu:
    cbz x6, save_result
    movi    d0, 0
    fmax    v16.4s, v16.4s, v0.4s
    fmax    v17.4s, v17.4s, v0.4s
    fmax    v18.4s, v18.4s, v0.4s
    fmax    v19.4s, v19.4s, v0.4s

save_result:
    str     q16, [x3]
    add     x0,x3,x4
    str     q17, [x0]
    add     x0,x0,x4
    str     q18, [x0]
    add x0,x0,x4
    str     q19, [x0]
    
	ret
                .end
