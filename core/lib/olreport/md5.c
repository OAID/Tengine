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
 * Author: jjzeng@openailab.com
 */

#include "md5.h"
#include <stdio.h>
#include <string.h>

//Magic initialization constants
#define MD5_INIT_STATE_0 0x67452301
#define MD5_INIT_STATE_1 0xefcdab89
#define MD5_INIT_STATE_2 0x98badcfe
#define MD5_INIT_STATE_3 0x10325476

//Constants for Transform routine.
#define MD5_S11  7
#define MD5_S12 12
#define MD5_S13 17
#define MD5_S14 22
#define MD5_S21  5
#define MD5_S22  9
#define MD5_S23 14
#define MD5_S24 20
#define MD5_S31  4
#define MD5_S32 11
#define MD5_S33 16
#define MD5_S34 23
#define MD5_S41  6
#define MD5_S42 10
#define MD5_S43 15
#define MD5_S44 21

//Transformation Constants - Round 1
#define MD5_T01  0xd76aa478 //Transformation Constant 1 
#define MD5_T02  0xe8c7b756 //Transformation Constant 2
#define MD5_T03  0x242070db //Transformation Constant 3
#define MD5_T04  0xc1bdceee //Transformation Constant 4
#define MD5_T05  0xf57c0faf //Transformation Constant 5
#define MD5_T06  0x4787c62a //Transformation Constant 6
#define MD5_T07  0xa8304613 //Transformation Constant 7
#define MD5_T08  0xfd469501 //Transformation Constant 8
#define MD5_T09  0x698098d8 //Transformation Constant 9
#define MD5_T10  0x8b44f7af //Transformation Constant 10
#define MD5_T11  0xffff5bb1 //Transformation Constant 11
#define MD5_T12  0x895cd7be //Transformation Constant 12
#define MD5_T13  0x6b901122 //Transformation Constant 13
#define MD5_T14  0xfd987193 //Transformation Constant 14
#define MD5_T15  0xa679438e //Transformation Constant 15
#define MD5_T16  0x49b40821 //Transformation Constant 16

//Transformation Constants - Round 2
#define MD5_T17  0xf61e2562 //Transformation Constant 17
#define MD5_T18  0xc040b340 //Transformation Constant 18
#define MD5_T19  0x265e5a51 //Transformation Constant 19
#define MD5_T20  0xe9b6c7aa //Transformation Constant 20
#define MD5_T21  0xd62f105d //Transformation Constant 21
#define MD5_T22  0x02441453 //Transformation Constant 22
#define MD5_T23  0xd8a1e681 //Transformation Constant 23
#define MD5_T24  0xe7d3fbc8 //Transformation Constant 24
#define MD5_T25  0x21e1cde6 //Transformation Constant 25
#define MD5_T26  0xc33707d6 //Transformation Constant 26
#define MD5_T27  0xf4d50d87 //Transformation Constant 27
#define MD5_T28  0x455a14ed //Transformation Constant 28
#define MD5_T29  0xa9e3e905 //Transformation Constant 29
#define MD5_T30  0xfcefa3f8 //Transformation Constant 30
#define MD5_T31  0x676f02d9 //Transformation Constant 31
#define MD5_T32  0x8d2a4c8a //Transformation Constant 32

//Transformation Constants - Round 3
#define MD5_T33  0xfffa3942 //Transformation Constant 33
#define MD5_T34  0x8771f681 //Transformation Constant 34
#define MD5_T35  0x6d9d6122 //Transformation Constant 35
#define MD5_T36  0xfde5380c //Transformation Constant 36
#define MD5_T37  0xa4beea44 //Transformation Constant 37
#define MD5_T38  0x4bdecfa9 //Transformation Constant 38
#define MD5_T39  0xf6bb4b60 //Transformation Constant 39
#define MD5_T40  0xbebfbc70 //Transformation Constant 40
#define MD5_T41  0x289b7ec6 //Transformation Constant 41
#define MD5_T42  0xeaa127fa //Transformation Constant 42
#define MD5_T43  0xd4ef3085 //Transformation Constant 43
#define MD5_T44  0x04881d05 //Transformation Constant 44
#define MD5_T45  0xd9d4d039 //Transformation Constant 45
#define MD5_T46  0xe6db99e5 //Transformation Constant 46
#define MD5_T47  0x1fa27cf8 //Transformation Constant 47
#define MD5_T48  0xc4ac5665 //Transformation Constant 48

//Transformation Constants - Round 4
#define MD5_T49  0xf4292244 //Transformation Constant 49
#define MD5_T50  0x432aff97 //Transformation Constant 50
#define MD5_T51  0xab9423a7 //Transformation Constant 51
#define MD5_T52  0xfc93a039 //Transformation Constant 52
#define MD5_T53  0x655b59c3 //Transformation Constant 53
#define MD5_T54  0x8f0ccc92 //Transformation Constant 54
#define MD5_T55  0xffeff47d //Transformation Constant 55
#define MD5_T56  0x85845dd1 //Transformation Constant 56
#define MD5_T57  0x6fa87e4f //Transformation Constant 57
#define MD5_T58  0xfe2ce6e0 //Transformation Constant 58
#define MD5_T59  0xa3014314 //Transformation Constant 59
#define MD5_T60  0x4e0811a1 //Transformation Constant 60
#define MD5_T61  0xf7537e82 //Transformation Constant 61
#define MD5_T62  0xbd3af235 //Transformation Constant 62
#define MD5_T63  0x2ad7d2bb //Transformation Constant 63
#define MD5_T64  0xeb86d391 //Transformation Constant 64

#define UINT32_TO_BYTE(n,b,i)\
{\
    (b)[(i)] = (unsigned char) ( ( (n) ) & 0xFF );\
    (b)[(i) + 1] = (unsigned char) ( ( (n) >>  8 ) & 0xFF );\
    (b)[(i) + 2] = (unsigned char) ( ( (n) >> 16 ) & 0xFF );\
    (b)[(i) + 3] = (unsigned char) ( ( (n) >> 24 ) & 0xFF );\
}

#define BYTE_TO_UINT32(n,b,i)\
{\
    (n) = ( (uint32_t) (b)[(i) ] ) \
        | ( (uint32_t) (b)[(i) + 1] <<  8 ) \
        | ( (uint32_t) (b)[(i) + 2] << 16 ) \
        | ( (uint32_t) (b)[(i) + 3] << 24 );\
}

#define R(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

#define FF(A,B,C,D,X,S,T) \
{\
    A += ( D ^ (  B & ( C ^ D ) ) ) + X + T; \
    A = R(A,S) + B; \
}

#define GG(A,B,C,D,X,S,T) \
{\
    A += ( C ^ ( D & ( B ^ C ) ) ) + X + T;\
    A = R(A,S) + B;\
}

#define HH(A,B,C,D,X,S,T) \
{\
    A += ( B ^ C ^ D ) + X + T;\
    A = R(A,S) + B;\
} 

#define II(A,B,C,D,X,S,T) \
{\
    A += ( C ^ ( B | ~D ) ) + X + T; \
    A = R(A,S)  + B; \
}

#define MD5FORAMT "%02X%02X%02X%02X"
#define MD5ARGS(args,idx) (args)[idx+0],(args)[idx+1],(args)[idx+2],(args)[idx+3]

void md5_to_string(const unsigned char md5[16],char* str_md5)
{
    sprintf(str_md5,MD5FORAMT MD5FORAMT MD5FORAMT MD5FORAMT,MD5ARGS(md5,0),MD5ARGS(md5,4),MD5ARGS(md5,8),MD5ARGS(md5,12));
}

int transform(md5_context* ctx,const unsigned char buff[64]);

int md5_start(md5_context* ctx)
{
    memset( ctx, 0, sizeof( md5_context ) );

    ctx->total[0] = 0;
    ctx->total[1] = 0;

    ctx->state[0] = MD5_INIT_STATE_0;
    ctx->state[1] = MD5_INIT_STATE_1;
    ctx->state[2] = MD5_INIT_STATE_2;
    ctx->state[3] = MD5_INIT_STATE_3;

    return 0;
}

int get_md5(const unsigned char* dat,size_t len,unsigned char md5[16])
{
    md5_context context;
    md5_start(&context);
    md5_update(&context,dat,len);
    md5_finish(&context,md5);
    return 0;
}

int md5_update(md5_context* ctx,const unsigned char* dat,size_t len)
{
    if( len == 0 )
    {
        return 0;
    }

    int ret;
    size_t fill;
    uint32_t left;

    left = ctx->total[0] & 0x3F;
    fill = 64 - left;

    ctx->total[0] += (uint32_t)len;
    ctx->total[0] &= 0xFFFFFFFF;

    if( ctx->total[0] < (uint32_t)len )
    {
        ctx->total[1]++;
    }

    if( left && len >= fill )
    {
        memcpy( (void *) (ctx->buffer + left), dat, fill );
        if( ( ret = transform( ctx, ctx->buffer ) ) != 0 )
        {
            return ret ;
        }

        dat += fill;
        len -= fill;
        left = 0;
    }

    while( len >= 64 )
    {
        if( ( ret = transform( ctx, dat ) ) != 0 )
        {
            return ret;
        }

        dat += 64;
        len -= 64;
    }

    if( len > 0 )
    {
        memcpy( (void *) (ctx->buffer + left), dat, len );
    }

    return 0;
}

int md5_finish(md5_context* ctx,unsigned char md5[16])
{
    int ret;
    uint32_t used;
    uint32_t high, low;

    
    //Add padding: 0x80 then 0x00 until 8 bytes remain for the length
    used = ctx->total[0] & 0x3F;
    ctx->buffer[used++] = 0x80;
    if( used <= 56 )
    {
        //Enough room for padding + length in current block
        memset( ctx->buffer + used, 0, 56 - used );
    }
    else
    {
        //need an extra block 
        memset( ctx->buffer + used, 0, 64 - used );

        if( ( ret = transform( ctx, ctx->buffer ) ) != 0 )
            return( ret );

        memset( ctx->buffer, 0, 56 );
    }

    //Add message length
    high = ( ctx->total[0] >> 29 )
         | ( ctx->total[1] <<  3 );
    low  = ( ctx->total[0] <<  3 );

    UINT32_TO_BYTE( low,  ctx->buffer, 56 );
    UINT32_TO_BYTE( high, ctx->buffer, 60 );

    if( ( ret = transform( ctx, ctx->buffer ) ) != 0 )
    {
        return ret ;
    }

    //set outpur
    UINT32_TO_BYTE( ctx->state[0], md5,  0 );
    UINT32_TO_BYTE( ctx->state[1], md5,  4 );
    UINT32_TO_BYTE( ctx->state[2], md5,  8 );
    UINT32_TO_BYTE( ctx->state[3], md5, 12 );

    return 0;
}

int transform(md5_context* ctx,const unsigned char buff[64])
{
    uint32_t X[16], a, b, c, d;
    int idx = 0;
    for(int ii=0;ii<16;++ii)
    {
        BYTE_TO_UINT32(X[ii],buff,idx);
        idx += 4;
    }

    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];

    //Perform Round 1 
	FF(a, b, c, d, X[ 0], MD5_S11, MD5_T01); 
	FF(d, a, b, c, X[ 1], MD5_S12, MD5_T02); 
	FF(c, d, a, b, X[ 2], MD5_S13, MD5_T03); 
	FF(b, c, d, a, X[ 3], MD5_S14, MD5_T04); 
	FF(a, b, c, d, X[ 4], MD5_S11, MD5_T05); 
	FF(d, a, b, c, X[ 5], MD5_S12, MD5_T06); 
	FF(c, d, a, b, X[ 6], MD5_S13, MD5_T07); 
	FF(b, c, d, a, X[ 7], MD5_S14, MD5_T08); 
	FF(a, b, c, d, X[ 8], MD5_S11, MD5_T09); 
	FF(d, a, b, c, X[ 9], MD5_S12, MD5_T10); 
	FF(c, d, a, b, X[10], MD5_S13, MD5_T11); 
	FF(b, c, d, a, X[11], MD5_S14, MD5_T12); 
	FF(a, b, c, d, X[12], MD5_S11, MD5_T13);
	FF(d, a, b, c, X[13], MD5_S12, MD5_T14); 
	FF(c, d, a, b, X[14], MD5_S13, MD5_T15); 
	FF(b, c, d, a, X[15], MD5_S14, MD5_T16); 

    //Perform Round 2
    GG(a, b, c, d, X[ 1], MD5_S21, MD5_T17); 
	GG(d, a, b, c, X[ 6], MD5_S22, MD5_T18); 
	GG(c, d, a, b, X[11], MD5_S23, MD5_T19); 
	GG(b, c, d, a, X[ 0], MD5_S24, MD5_T20); 
	GG(a, b, c, d, X[ 5], MD5_S21, MD5_T21); 
	GG(d, a, b, c, X[10], MD5_S22, MD5_T22); 
	GG(c, d, a, b, X[15], MD5_S23, MD5_T23); 
	GG(b, c, d, a, X[ 4], MD5_S24, MD5_T24); 
	GG(a, b, c, d, X[ 9], MD5_S21, MD5_T25); 
	GG(d, a, b, c, X[14], MD5_S22, MD5_T26); 
	GG(c, d, a, b, X[ 3], MD5_S23, MD5_T27); 
	GG(b, c, d, a, X[ 8], MD5_S24, MD5_T28); 
	GG(a, b, c, d, X[13], MD5_S21, MD5_T29); 
	GG(d, a, b, c, X[ 2], MD5_S22, MD5_T30); 
	GG(c, d, a, b, X[ 7], MD5_S23, MD5_T31); 
	GG(b, c, d, a, X[12], MD5_S24, MD5_T32);

    //Perform Round 3 
	HH(a, b, c, d, X[ 5], MD5_S31, MD5_T33); 
	HH(d, a, b, c, X[ 8], MD5_S32, MD5_T34); 
	HH(c, d, a, b, X[11], MD5_S33, MD5_T35); 
	HH(b, c, d, a, X[14], MD5_S34, MD5_T36); 
	HH(a, b, c, d, X[ 1], MD5_S31, MD5_T37); 
	HH(d, a, b, c, X[ 4], MD5_S32, MD5_T38); 
	HH(c, d, a, b, X[ 7], MD5_S33, MD5_T39); 
	HH(b, c, d, a, X[10], MD5_S34, MD5_T40); 
	HH(a, b, c, d, X[13], MD5_S31, MD5_T41); 
	HH(d, a, b, c, X[ 0], MD5_S32, MD5_T42); 
	HH(c, d, a, b, X[ 3], MD5_S33, MD5_T43); 
	HH(b, c, d, a, X[ 6], MD5_S34, MD5_T44); 
	HH(a, b, c, d, X[ 9], MD5_S31, MD5_T45); 
	HH(d, a, b, c, X[12], MD5_S32, MD5_T46); 
	HH(c, d, a, b, X[15], MD5_S33, MD5_T47); 
	HH(b, c, d, a, X[ 2], MD5_S34, MD5_T48); 

    //Perform Round 4 
	II(a, b, c, d, X[ 0], MD5_S41, MD5_T49); 
	II(d, a, b, c, X[ 7], MD5_S42, MD5_T50); 
	II(c, d, a, b, X[14], MD5_S43, MD5_T51); 
	II(b, c, d, a, X[ 5], MD5_S44, MD5_T52); 
	II(a, b, c, d, X[12], MD5_S41, MD5_T53); 
	II(d, a, b, c, X[ 3], MD5_S42, MD5_T54); 
	II(c, d, a, b, X[10], MD5_S43, MD5_T55); 
	II(b, c, d, a, X[ 1], MD5_S44, MD5_T56); 
	II(a, b, c, d, X[ 8], MD5_S41, MD5_T57); 
	II(d, a, b, c, X[15], MD5_S42, MD5_T58); 
	II(c, d, a, b, X[ 6], MD5_S43, MD5_T59); 
	II(b, c, d, a, X[13], MD5_S44, MD5_T60); 
	II(a, b, c, d, X[ 4], MD5_S41, MD5_T61); 
	II(d, a, b, c, X[11], MD5_S42, MD5_T62); 
	II(c, d, a, b, X[ 2], MD5_S43, MD5_T63); 
	II(b, c, d, a, X[ 9], MD5_S44, MD5_T64);

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;

    return 0;
}
