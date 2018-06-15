#define FUNCTION(name) \
        .text ;\
        .align  4 ;\
        .global name ;\
        .type   name, %function ;\
name:


#define ENDFUNCTION(name)

#define TWO_MLA(INST,type)  \
                  INST v16.## type,v17.## type,v18.## type;   \
                  INST v20.## type,v21.## type,v22.## type;  


#define FOUR_MLA(INST,type)  \
                  INST v0.## type,v1.## type,v2.## type ;   \
                  INST v4.## type,v5.## type,v6.## type ;   \
                  INST v8.## type,v9.## type,v10.## type ;  \
                  INST v12.## type,v13.## type,v14.## type ;


#define TEN_MLA(INST,type)    FOUR_MLA(INST,type) FOUR_MLA(INST,type) TWO_MLA(INST,type)
#define FIFTY_MLA(INST,type)   TEN_MLA(INST,type) TEN_MLA(INST,type) TEN_MLA(INST,type) TEN_MLA(INST,type) TEN_MLA(INST,type)
#define HUNDRED_MLA(INST,type) FIFTY_MLA(INST,type) FIFTY_MLA(INST,type)
#define TWO_HUNDRED_MLA(INST,type) HUNDRED_MLA(INST,type) HUNDRED_MLA(INST,type)



#define TWO_HUNDRED_MLA_4S TWO_HUNDRED_MLA(mla,4s)
#define TEN_MLA_4S TEN_MLA(mla,4s)

FUNCTION(test_mla_4s)

   mov x1,0
1:   
   TWO_HUNDRED_MLA_4S
   TWO_HUNDRED_MLA_4S
   TWO_HUNDRED_MLA_4S
   TWO_HUNDRED_MLA_4S
   TWO_HUNDRED_MLA_4S
   add x1,x1,1  
   cmp x0,x1
   TEN_MLA_4S
   TEN_MLA_4S
   bne 1b
   ret

ENDFUNCTION(test_mla_4s)

#define TWO_HUNDRED_MLA_16B TWO_HUNDRED_MLA(mla,16b)
#define TEN_MLA_16B TEN_MLA(mla,16b)

FUNCTION(test_mla_16b)

   mov x1,0
1:   
   TWO_HUNDRED_MLA_16B
   TWO_HUNDRED_MLA_16B
   TWO_HUNDRED_MLA_16B
   TWO_HUNDRED_MLA_16B
   TWO_HUNDRED_MLA_16B
   add x1,x1,1  
   cmp x0,x1
   TEN_MLA_16B
   TEN_MLA_16B
   bne 1b
   ret

ENDFUNCTION(test_mla_16b)

#define TWO_HUNDRED_MLA_8H TWO_HUNDRED_MLA(mla,8h)
#define TEN_MLA_8H   TEN_MLA(mla,8h)


FUNCTION(test_mla_8h)

   mov x1,0
1:
   TWO_HUNDRED_MLA_8H
   TWO_HUNDRED_MLA_8H
   TWO_HUNDRED_MLA_8H
   TWO_HUNDRED_MLA_8H
   TWO_HUNDRED_MLA_8H
   add x1,x1,1
   cmp x0,x1
   TEN_MLA_8H
   TEN_MLA_8H
   bne 1b
   ret
ENDFUNCTION(test_mla_8h)



#define TEN_MLA_IDX(INST,type)  \
                  INST v0.## type,v1.## type,v8.## type[0] ;   \
                  INST v4.## type,v5.## type,v12.## type[1] ;   \
                  INST v8.## type,v9.## type,v16.## type[2] ;  \
                  INST v12.## type,v13.## type,v20.## type[3] ;\
                  INST v16.## type,v17.## type,v24.## type[0]; \
                  INST v20.## type,v21.## type,v28.## type[1]; \
                  INST v24.## type,v25.## type,v3.## type[1]; \
                  INST v28.## type,v29.## type,v15.## type[2]; \
                  INST v3.## type, v7.## type, v0.## type;    \
                  INST v15.## type, v19.## type, v4.## type;  


#define FIFTY_MLA_IDX(INST,type)   TEN_MLA_IDX(INST,type) TEN_MLA_IDX(INST,type) TEN_MLA_IDX(INST,type) TEN_MLA_IDX(INST,type) TEN_MLA_IDX(INST,type)
#define HUNDRED_MLA_IDX(INST,type) FIFTY_MLA_IDX(INST,type) FIFTY_MLA_IDX(INST,type)
#define TWO_HUNDRED_MLA_IDX(INST,type) HUNDRED_MLA_IDX(INST,type) HUNDRED_MLA_IDX(INST,type)

#define TWO_HUNDRED_FMLA_4S TWO_HUNDRED_MLA_IDX(fmla,4s)
#define TEN_FMLA_4S   TEN_MLA_IDX(fmla,4s)

FUNCTION(test_fmla_4s)

   mov x1,0
1:   
   TWO_HUNDRED_FMLA_4S
   TWO_HUNDRED_FMLA_4S
   TWO_HUNDRED_FMLA_4S
   TWO_HUNDRED_FMLA_4S
   TWO_HUNDRED_FMLA_4S
   add x1,x1,1  
   cmp x0,x1
   TEN_FMLA_4S
   TEN_FMLA_4S
   bne 1b
   ret

ENDFUNCTION(test_fmla_4s)

