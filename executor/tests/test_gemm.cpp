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
#include <iostream>

#include <cblas.h>
#include <unistd.h>
#include <cstring>

#include "prof_utils.hpp"

using namespace TEngine;

extern "C" {
void blas_gemm_4x4(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int , int , int, float,
                             float *, int, float *, int, float, float *, int);

void blas_gemm_16x4(CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int , int , int, float,
                             float *, int, float *, int, float, float *, int);

extern int sgemm_kernel_16x4 (long, long, long, float,  float  *, float  *, float  *, long);
extern int sgemm_kernel_4x4 (long, long, long, float,  float  *, float  *, float  *, long);

}

#if 1
   int N=32;
   int K=288;
   int M=12544;
#else
   int N=256;
   int K=256;
   int M=1024;
#endif

int run_test(int rep)
{

   float * A=(float *)std::malloc(N*K*sizeof(float));
   float * B=(float *)std::malloc(M*K*sizeof(float));
   float * C=(float *)std::malloc(M*N*sizeof(float));

   std::memset(A,0x1,N*K*sizeof(float));
   std::memset(B,0x1,M*K*sizeof(float));
   std::memset(C,0x1,N*M*sizeof(float));

   float ops=1.0*N*M*K*2;

   unsigned long start=get_cur_time();

   for(int i=0;i<rep;i++)
       cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,K,1.0f,
                           A,K,B,M, 1.0f,C,M);

   unsigned long end=get_cur_time();

   std::printf("cblas used: %lu us, rate: %.2f Mops\n", end-start, (ops*rep)/(end-start));

   /***** */

   float * packing_A=(float *)std::malloc(N*K*sizeof(float));
   float * packing_B=(float *)std::malloc(M*K*sizeof(float));


    std::memset(packing_A,0x1,N*K*sizeof(float));
    std::memset(packing_B,0x1,M*K*sizeof(float));
   
    start=get_cur_time();

    for(int i=0;i<rep;i++)
    {
       //sgemm_packingA(K,N,A,K,packing_A);
      // sgemm_packingB(K,M,B,M,packing_B);
      // sgemm_kernel_16x4(M,N,K,1.0,packing_B,packing_A,C,M);

       blas_gemm_16x4(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,M,K,1.0f,
                           A,K,B,M, 1.0f,C,M);

    }

    end=get_cur_time();

   std::printf("porting used: %lu us, rate: %.2f Mops\n", end-start, (ops*rep)/(end-start));

   return 0;
}

   

int main(int argc, char * argv[])
{
    int res;
    int rep=1;

    while((res=getopt(argc,argv,"r:M:K:N:"))!=-1)
    {
        switch(res)
        {
        case 'r':
            rep=strtoul(optarg,NULL,10);
            break;
        case 'M':
            M=strtoul(optarg,NULL,10);
            break;
        case 'K':
            K=strtoul(optarg,NULL,10);
            break;
        case 'N':
            N=strtoul(optarg,NULL,10);
            break;
        default:
            break;
        }
    }

    run_test(rep);

    return 1;
}




