#include <stdlib.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#if __AVX__
static void dwconv3x3s1d1(int inc, int inw, int inh, int outw, int outh, float* kernel_data, float* const img_data, float* const bias_data, float* const output, bool have_biases)
{
    int inwh = inw * inh;
    int outwh = outw * outh;
    int channel_count = inc >> 3;
    int channel_remain = inc - (channel_count << 3);
    //generate the image tmp
    float* img_tmp = (float*) malloc(8 * inwh * (channel_count + 1) * sizeof(float)); 
    float* kernel_tmp = (float*) malloc(8 * 9 * (channel_count + 1) * sizeof(float));
    float* bias_tmp = (float*) malloc(8 * (channel_count + 1) * sizeof(float));
    {
        for(int i = 0; i < channel_count; i++)
        {
            int ii = i * 8;
            const float* k0 = img_data + (ii + 0) * inwh;
            const float* k1 = img_data + (ii + 1) * inwh;
            const float* k2 = img_data + (ii + 2) * inwh;
            const float* k3 = img_data + (ii + 3) * inwh;
            const float* k4 = img_data + (ii + 4) * inwh;
            const float* k5 = img_data + (ii + 5) * inwh;
            const float* k6 = img_data + (ii + 6) * inwh;
            const float* k7 = img_data + (ii + 7) * inwh;
        
            const float* f0 = kernel_data + (ii + 0) * 9;
            const float* f1 = kernel_data + (ii + 1) * 9;
            const float* f2 = kernel_data + (ii + 2) * 9;
            const float* f3 = kernel_data + (ii + 3) * 9;
            const float* f4 = kernel_data + (ii + 4) * 9;
            const float* f5 = kernel_data + (ii + 5) * 9;
            const float* f6 = kernel_data + (ii + 6) * 9;
            const float* f7 = kernel_data + (ii + 7) * 9;

            const float* b0 = bias_data + (ii + 0);
            const float* b1 = bias_data + (ii + 1);
            const float* b2 = bias_data + (ii + 2);
            const float* b3 = bias_data + (ii + 3);
            const float* b4 = bias_data + (ii + 4);
            const float* b5 = bias_data + (ii + 5);
            const float* b6 = bias_data + (ii + 6);
            const float* b7 = bias_data + (ii + 7);

            float* tmp0 = img_tmp + ii * inwh;
            float* tmp1 = kernel_tmp + ii * 9;
            float* tmp2 = bias_tmp + ii;
            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
                tmp0[4] = k4[0];
                tmp0[5] = k5[0];
                tmp0[6] = k6[0];
                tmp0[7] = k7[0];
            
                tmp0 += 8;

                k0++;
                k1++;
                k2++;
                k3++;
                k4++;
                k5++;
                k6++;
                k7++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                tmp1[4] = f4[0];
                tmp1[5] = f5[0];
                tmp1[6] = f6[0];
                tmp1[7] = f7[0];

                tmp1 += 8;
                f0++;
                f1++;
                f2++;
                f3++;
                f4++;
                f5++;
                f6++;
                f7++;
            }
            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
                tmp2[4] = b4[0];
                tmp2[5] = b5[0];
                tmp2[6] = b6[0];
                tmp2[7] = b7[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
                tmp2[4] = 0;
                tmp2[5] = 0;
                tmp2[6] = 0;
                tmp2[7] = 0;
            }
        } 
        int i = 0;
        for(; i + 3 < channel_remain; i += 4)
        {
            int ii = channel_count * 8 + i;
            float* k0 = img_data + (ii + 0) * inwh;
            float* k1 = img_data + (ii + 1) * inwh;
            float* k2 = img_data + (ii + 2) * inwh;
            float* k3 = img_data + (ii + 3) * inwh;
        
            float* f0 = kernel_data + (ii + 0) * 9;
            float* f1 = kernel_data + (ii + 1) * 9;
            float* f2 = kernel_data + (ii + 2) * 9;
            float* f3 = kernel_data + (ii + 3) * 9;

            float* b0 = bias_data + (ii + 0);
            float* b1 = bias_data + (ii + 1);
            float* b2 = bias_data + (ii + 2);
            float* b3 = bias_data + (ii + 3);

            float* tmp0 = img_tmp + channel_count * 8 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 8 * 9;
            float* tmp2 = bias_tmp + ii;
            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
            
                tmp0 += 8;

                k0++;
                k1++;
                k2++;
                k3++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                
                tmp1 += 8;
                f0++;
                f1++;
                f2++;
                f3++;
            }
            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
            }
        }

        for(; i < channel_remain; i++)
        {
            int ii = channel_count * 8 + i;
            float* k0 = img_data + ii * inwh;
            float* f0 = kernel_data + ii * 9;
            float* b0 = bias_data + ii; 

            float* tmp0 = img_tmp + channel_count * 8 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 8 * 9;
            float* tmp2 = bias_tmp + channel_count * 8;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[i] = k0[0];

                tmp0 += 8;
                k0++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[i] = f0[0];
                                
                tmp1 += 8;
                f0++;
            }
            if(have_biases)
            {
                tmp2[i] = b0[0];
            }
            else
            {
                tmp2[i] = 0;
            }
            
        }
    }

    float* output_tmp = (float*) malloc(outwh * (channel_count + 1) * 8 * sizeof(float));
    for(int c = 0; c < channel_count+1; c++)
    {
        float* ktmp = kernel_tmp + c * 8 * 9;
        float* btmp = bias_tmp + c * 8;
        for(int i = 0; i < outh; i++)
        {
            int j = 0;

            float* itmp0 = img_tmp + c * 8 * inwh + 8 * i * inw;
            float* itmp1 = img_tmp + c * 8 * inwh + 8 * (i + 1) * inw;
            float* itmp2 = img_tmp + c * 8 * inwh + 8 * (i + 2) * inw;
            float* otmp = output_tmp + c * 8 * outwh+ 8 * i * outw;
            for(;j + 7 < outw;j += 8)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);
                __m256 _sum1 = _mm256_loadu_ps(btmp);
                __m256 _sum2 = _mm256_loadu_ps(btmp);
                __m256 _sum3 = _mm256_loadu_ps(btmp);
                __m256 _sum4 = _mm256_loadu_ps(btmp);
                __m256 _sum5 = _mm256_loadu_ps(btmp);
                __m256 _sum6 = _mm256_loadu_ps(btmp);
                __m256 _sum7 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);
                __m256 _va3 = _mm256_loadu_ps(itmp0+24);
                __m256 _va4 = _mm256_loadu_ps(itmp0+32);
                __m256 _va5 = _mm256_loadu_ps(itmp0+40);
                __m256 _va6 = _mm256_loadu_ps(itmp0+48);
                __m256 _va7 = _mm256_loadu_ps(itmp0+56);
                __m256 _va8 = _mm256_loadu_ps(itmp0+64);
                __m256 _va9 = _mm256_loadu_ps(itmp0+72);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum4 = _mm256_fmadd_ps(_va4, _vb0, _sum4);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);
                _sum5 = _mm256_fmadd_ps(_va5, _vb0, _sum5);
                _sum4 = _mm256_fmadd_ps(_va5, _vb1, _sum4);
                _sum5 = _mm256_fmadd_ps(_va6, _vb1, _sum5);
                _sum4 = _mm256_fmadd_ps(_va6, _vb2, _sum4);
                _sum6 = _mm256_fmadd_ps(_va6, _vb0, _sum6);
                _sum7 = _mm256_fmadd_ps(_va7, _vb0, _sum7);
                _sum5 = _mm256_fmadd_ps(_va7, _vb2, _sum5);
                _sum6 = _mm256_fmadd_ps(_va7, _vb1, _sum6);
                _sum7 = _mm256_fmadd_ps(_va8, _vb1, _sum7);
                _sum6 = _mm256_fmadd_ps(_va8, _vb2, _sum6);
                _sum7 = _mm256_fmadd_ps(_va9, _vb2, _sum7);
                
                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);
                _va3 = _mm256_loadu_ps(itmp1+24);
                _va4 = _mm256_loadu_ps(itmp1+32);
                _va5 = _mm256_loadu_ps(itmp1+40);
                _va6 = _mm256_loadu_ps(itmp1+48);
                _va7 = _mm256_loadu_ps(itmp1+56);
                _va8 = _mm256_loadu_ps(itmp1+64);
                _va9 = _mm256_loadu_ps(itmp1+72);
                
                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum4 = _mm256_fmadd_ps(_va4, _vb0, _sum4);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);
                _sum5 = _mm256_fmadd_ps(_va5, _vb0, _sum5);
                _sum4 = _mm256_fmadd_ps(_va5, _vb1, _sum4);
                _sum5 = _mm256_fmadd_ps(_va6, _vb1, _sum5);
                _sum4 = _mm256_fmadd_ps(_va6, _vb2, _sum4);
                _sum6 = _mm256_fmadd_ps(_va6, _vb0, _sum6);
                _sum7 = _mm256_fmadd_ps(_va7, _vb0, _sum7);
                _sum5 = _mm256_fmadd_ps(_va7, _vb2, _sum5);
                _sum6 = _mm256_fmadd_ps(_va7, _vb1, _sum6);
                _sum7 = _mm256_fmadd_ps(_va8, _vb1, _sum7);
                _sum6 = _mm256_fmadd_ps(_va8, _vb2, _sum6);
                _sum7 = _mm256_fmadd_ps(_va9, _vb2, _sum7);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);
                _va3 = _mm256_loadu_ps(itmp2+24);
                _va4 = _mm256_loadu_ps(itmp2+32);
                _va5 = _mm256_loadu_ps(itmp2+40);
                _va6 = _mm256_loadu_ps(itmp2+48);
                _va7 = _mm256_loadu_ps(itmp2+56);
                _va8 = _mm256_loadu_ps(itmp2+64);
                _va9 = _mm256_loadu_ps(itmp2+72);
                
                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum4 = _mm256_fmadd_ps(_va4, _vb0, _sum4);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);
                _sum5 = _mm256_fmadd_ps(_va5, _vb0, _sum5);
                _sum4 = _mm256_fmadd_ps(_va5, _vb1, _sum4);
                _sum5 = _mm256_fmadd_ps(_va6, _vb1, _sum5);
                _sum4 = _mm256_fmadd_ps(_va6, _vb2, _sum4);
                _sum6 = _mm256_fmadd_ps(_va6, _vb0, _sum6);
                _sum7 = _mm256_fmadd_ps(_va7, _vb0, _sum7);
                _sum5 = _mm256_fmadd_ps(_va7, _vb2, _sum5);
                _sum6 = _mm256_fmadd_ps(_va7, _vb1, _sum6);
                _sum7 = _mm256_fmadd_ps(_va8, _vb1, _sum7);
                _sum6 = _mm256_fmadd_ps(_va8, _vb2, _sum6);
                _sum7 = _mm256_fmadd_ps(_va9, _vb2, _sum7);

                _mm256_storeu_ps(otmp, _sum0);
                _mm256_storeu_ps(otmp+8, _sum1);
                _mm256_storeu_ps(otmp+16, _sum2);
                _mm256_storeu_ps(otmp+24, _sum3);
                _mm256_storeu_ps(otmp+32, _sum4);
                _mm256_storeu_ps(otmp+40, _sum5);
                _mm256_storeu_ps(otmp+48, _sum6);
                _mm256_storeu_ps(otmp+56, _sum7);

                itmp0 += 64;
                itmp1 += 64;
                itmp2 += 64;
                otmp  += 64;
            }

            for(;j + 3 < outw;j += 4)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);
                __m256 _sum1 = _mm256_loadu_ps(btmp);
                __m256 _sum2 = _mm256_loadu_ps(btmp);
                __m256 _sum3 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);
                __m256 _va3 = _mm256_loadu_ps(itmp0+24);
                __m256 _va4 = _mm256_loadu_ps(itmp0+32);
                __m256 _va5 = _mm256_loadu_ps(itmp0+40);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);

                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);
                _va3 = _mm256_loadu_ps(itmp1+24);
                _va4 = _mm256_loadu_ps(itmp1+32);
                _va5 = _mm256_loadu_ps(itmp1+40);
                
                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);
                _va3 = _mm256_loadu_ps(itmp2+24);
                _va4 = _mm256_loadu_ps(itmp2+32);
                _va5 = _mm256_loadu_ps(itmp2+40);
                
                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum2 = _mm256_fmadd_ps(_va2, _vb0, _sum2);
                _sum3 = _mm256_fmadd_ps(_va3, _vb0, _sum3);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va3, _vb1, _sum2);
                _sum3 = _mm256_fmadd_ps(_va4, _vb1, _sum3);
                _sum2 = _mm256_fmadd_ps(_va4, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va5, _vb2, _sum3);

                _mm256_storeu_ps(otmp, _sum0);
                _mm256_storeu_ps(otmp+8, _sum1);
                _mm256_storeu_ps(otmp+16, _sum2);
                _mm256_storeu_ps(otmp+24, _sum3);

                itmp0 += 32;
                itmp1 += 32;
                itmp2 += 32;
                otmp  += 32;
            }

            for(;j + 1 < outw;j += 2)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);
                __m256 _sum1 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);
                __m256 _va3 = _mm256_loadu_ps(itmp0+24);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);

                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);
                _va3 = _mm256_loadu_ps(itmp1+24);

                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);
                _va3 = _mm256_loadu_ps(itmp2+24);

                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum1 = _mm256_fmadd_ps(_va1, _vb0, _sum1);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb1, _sum1);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va3, _vb2, _sum1);

                _mm256_storeu_ps(otmp, _sum0);
                _mm256_storeu_ps(otmp+8, _sum1);

                itmp0 += 16;
                itmp1 += 16;
                itmp2 += 16;
                otmp  += 16;
            }

            for(;j < outw;j++)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);

                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);

                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _mm256_storeu_ps(otmp, _sum0);

                itmp0 += 8;
                itmp1 += 8;
                itmp2 += 8;
                otmp  += 8;
            }
        }
    }
    //load_data
    {
        for(int i = 0; i < channel_count; i++)
        {
            float* otmp = output_tmp + i * 8 * outwh;

            float* tmp0 = output + i * 8 * outwh;
            float* tmp1 = output + i * 8 * outwh + 1 * outwh;
            float* tmp2 = output + i * 8 * outwh + 2 * outwh;
            float* tmp3 = output + i * 8 * outwh + 3 * outwh;
            float* tmp4 = output + i * 8 * outwh + 4 * outwh;
            float* tmp5 = output + i * 8 * outwh + 5 * outwh;
            float* tmp6 = output + i * 8 * outwh + 6 * outwh;
            float* tmp7 = output + i * 8 * outwh + 7 * outwh;
            for(int i = 0; i < outwh; i++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                tmp4[0] = otmp[4];
                tmp5[0] = otmp[5];
                tmp6[0] = otmp[6];
                tmp7[0] = otmp[7];
                otmp += 8;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
                tmp4++;
                tmp5++;
                tmp6++;
                tmp7++;
            }
        }
        int i = 0;
        for(; i + 3 < channel_remain; i += 4)
        {
            int ii = channel_count * 8 + i;
            float* otmp = output_tmp + ii * outwh;

            float* tmp0 = output + ii * outwh;
            float* tmp1 = output + ii * outwh + 1 * outwh;
            float* tmp2 = output + ii * outwh + 2 * outwh;
            float* tmp3 = output + ii * outwh + 3 * outwh;
            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                
                otmp += 8;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
            }
        }

        for(; i < channel_remain; i++)
        {
            int ii = channel_count * 8 + i;
            float* otmp = output_tmp + channel_count * 8 * outwh;

            float* tmp0 = output + ii * outwh;

            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[i];

                otmp += 8;
                tmp0++;
            }
        }
    }
    free(output_tmp);
    free(img_tmp);
    free(kernel_tmp);
    free(bias_tmp);
}

static void dwconv3x3s2d1(int inc, int inw, int inh, int outw, int outh, float* kernel_data, float* const img_data, float* const bias_data, float* const output, bool have_biases)
{
    int inwh = inw * inh;
    int outwh = outw * outh;
    int channel_count = inc >> 3;
    int channel_remain = inc - (channel_count << 3);
    //generate the image tmp
    float* img_tmp = (float*) malloc(8 * inwh * (channel_count + 1) * sizeof(float)); 
    float* kernel_tmp = (float*) malloc(8 * 9 * (channel_count + 1) * sizeof(float));
    float* bias_tmp = (float*) malloc(8 * (channel_count + 1) * sizeof(float));
    {
        for(int i = 0; i < channel_count; i++)
        {
            int ii = i * 8;
            const float* k0 = img_data + (ii + 0) * inwh;
            const float* k1 = img_data + (ii + 1) * inwh;
            const float* k2 = img_data + (ii + 2) * inwh;
            const float* k3 = img_data + (ii + 3) * inwh;
            const float* k4 = img_data + (ii + 4) * inwh;
            const float* k5 = img_data + (ii + 5) * inwh;
            const float* k6 = img_data + (ii + 6) * inwh;
            const float* k7 = img_data + (ii + 7) * inwh;
        
            const float* f0 = kernel_data + (ii + 0) * 9;
            const float* f1 = kernel_data + (ii + 1) * 9;
            const float* f2 = kernel_data + (ii + 2) * 9;
            const float* f3 = kernel_data + (ii + 3) * 9;
            const float* f4 = kernel_data + (ii + 4) * 9;
            const float* f5 = kernel_data + (ii + 5) * 9;
            const float* f6 = kernel_data + (ii + 6) * 9;
            const float* f7 = kernel_data + (ii + 7) * 9;

            const float* b0 = bias_data + (ii + 0);
            const float* b1 = bias_data + (ii + 1);
            const float* b2 = bias_data + (ii + 2);
            const float* b3 = bias_data + (ii + 3);
            const float* b4 = bias_data + (ii + 4);
            const float* b5 = bias_data + (ii + 5);
            const float* b6 = bias_data + (ii + 6);
            const float* b7 = bias_data + (ii + 7);

            float* tmp0 = img_tmp + ii * inwh;
            float* tmp1 = kernel_tmp + ii * 9;
            float* tmp2 = bias_tmp + ii;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
                tmp0[4] = k4[0];
                tmp0[5] = k5[0];
                tmp0[6] = k6[0];
                tmp0[7] = k7[0];
        
                tmp0 += 8;

                k0++;
                k1++;
                k2++;
                k3++;
                k4++;
                k5++;
                k6++;
                k7++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                tmp1[4] = f4[0];
                tmp1[5] = f5[0];
                tmp1[6] = f6[0];
                tmp1[7] = f7[0];

                tmp1 += 8;
                f0++;
                f1++;
                f2++;
                f3++;
                f4++;
                f5++;
                f6++;
                f7++;
            }
            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
                tmp2[4] = b4[0];
                tmp2[5] = b5[0];
                tmp2[6] = b6[0];
                tmp2[7] = b7[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
                tmp2[4] = 0;
                tmp2[5] = 0;
                tmp2[6] = 0;
                tmp2[7] = 0;
            }
        } 
        int i = 0;
        for(; i + 3 < channel_remain; i += 4)
        {
            int ii = channel_count * 8 + i;
            float* k0 = img_data + (ii + 0) * inwh;
            float* k1 = img_data + (ii + 1) * inwh;
            float* k2 = img_data + (ii + 2) * inwh;
            float* k3 = img_data + (ii + 3) * inwh;
        
            float* f0 = kernel_data + (ii + 0) * 9;
            float* f1 = kernel_data + (ii + 1) * 9;
            float* f2 = kernel_data + (ii + 2) * 9;
            float* f3 = kernel_data + (ii + 3) * 9;

            float* b0 = bias_data + (ii + 0);
            float* b1 = bias_data + (ii + 1);
            float* b2 = bias_data + (ii + 2);
            float* b3 = bias_data + (ii + 3);

            float* tmp0 = img_tmp + channel_count * 8 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 8 * 9;
            float* tmp2 = bias_tmp + ii;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
            
                tmp0 += 8;

                k0++;
                k1++;
                k2++;
                k3++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                
                tmp1 += 8;
                f0++;
                f1++;
                f2++;
                f3++;
            }

            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
            }
        }

        for(; i < channel_remain; i++)
        {
            int ii = channel_count * 8 + i;
            float* k0 = img_data + ii * inwh;
            float* f0 = kernel_data + ii * 9;
            float* b0 = bias_data + ii; 

            float* tmp0 = img_tmp + channel_count * 8 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 8 * 9;
            float* tmp2 = bias_tmp + channel_count * 8;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[i] = k0[0];

                tmp0 += 8;
                k0++;
            }
            for(int j = 0; j < 9; j++)
            {
                tmp1[i] = f0[0];
                                
                tmp1 += 8;
                f0++;
            }

            if(have_biases)
            {
                tmp2[i] = b0[0];
            }
            else
            {
                tmp2[i] = 0;
            }
        }
    }

    float* output_tmp = (float*) malloc(outwh * (channel_count + 1) * 8 * sizeof(float));
    for(int c = 0; c < channel_count+1; c++)
    {
        float* ktmp = kernel_tmp + c * 8 * 9;
        float* btmp = bias_tmp + c * 8;
        for(int i = 0; i < outh; i++)
        {
            int j = 0;

            float* itmp0 = img_tmp + c * 8 * inwh + 8 * i * 2 * inw;
            float* itmp1 = img_tmp + c * 8 * inwh + 8 * (i * 2 + 1) * inw;
            float* itmp2 = img_tmp + c * 8 * inwh + 8 * (i * 2 + 2) * inw;
            float* otmp = output_tmp + c * 8 * outwh+ 8 * i * outw;
            for(;j + 3 < outw;j += 4)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);
                __m256 _sum1 = _mm256_loadu_ps(btmp);
                __m256 _sum2 = _mm256_loadu_ps(btmp);
                __m256 _sum3 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);
                __m256 _va3 = _mm256_loadu_ps(itmp0+24);
                __m256 _va4 = _mm256_loadu_ps(itmp0+32);
                __m256 _va5 = _mm256_loadu_ps(itmp0+40);
                __m256 _va6 = _mm256_loadu_ps(itmp0+48);
                __m256 _va7 = _mm256_loadu_ps(itmp0+56);
                __m256 _va8 = _mm256_loadu_ps(itmp0+64);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va4, _vb0, _sum2);
                _sum2 = _mm256_fmadd_ps(_va5, _vb1, _sum2);
                _sum2 = _mm256_fmadd_ps(_va6, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va6, _vb0, _sum3);
                _sum3 = _mm256_fmadd_ps(_va7, _vb1, _sum3);
                _sum3 = _mm256_fmadd_ps(_va8, _vb2, _sum3);
                
                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);
                _va3 = _mm256_loadu_ps(itmp1+24);
                _va4 = _mm256_loadu_ps(itmp1+32);
                _va5 = _mm256_loadu_ps(itmp1+40);
                _va6 = _mm256_loadu_ps(itmp1+48);
                _va7 = _mm256_loadu_ps(itmp1+56);
                _va8 = _mm256_loadu_ps(itmp1+64);
                
                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va4, _vb0, _sum2);
                _sum2 = _mm256_fmadd_ps(_va5, _vb1, _sum2);
                _sum2 = _mm256_fmadd_ps(_va6, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va6, _vb0, _sum3);
                _sum3 = _mm256_fmadd_ps(_va7, _vb1, _sum3);
                _sum3 = _mm256_fmadd_ps(_va8, _vb2, _sum3);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);
                _va3 = _mm256_loadu_ps(itmp2+24);
                _va4 = _mm256_loadu_ps(itmp2+32);
                _va5 = _mm256_loadu_ps(itmp2+40);
                _va6 = _mm256_loadu_ps(itmp2+48);
                _va7 = _mm256_loadu_ps(itmp2+56);
                _va8 = _mm256_loadu_ps(itmp2+64);
                
                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);
                _sum2 = _mm256_fmadd_ps(_va4, _vb0, _sum2);
                _sum2 = _mm256_fmadd_ps(_va5, _vb1, _sum2);
                _sum2 = _mm256_fmadd_ps(_va6, _vb2, _sum2);
                _sum3 = _mm256_fmadd_ps(_va6, _vb0, _sum3);
                _sum3 = _mm256_fmadd_ps(_va7, _vb1, _sum3);
                _sum3 = _mm256_fmadd_ps(_va8, _vb2, _sum3);

                _mm256_storeu_ps(otmp, _sum0);
                _mm256_storeu_ps(otmp+8, _sum1);
                _mm256_storeu_ps(otmp+16, _sum2);
                _mm256_storeu_ps(otmp+24, _sum3);

                itmp0 += 64;
                itmp1 += 64;
                itmp2 += 64;
                otmp  += 32;
            }

            for(;j + 1 < outw;j += 2)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);
                __m256 _sum1 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);
                __m256 _va3 = _mm256_loadu_ps(itmp0+24);
                __m256 _va4 = _mm256_loadu_ps(itmp0+32);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);

                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);
                _va3 = _mm256_loadu_ps(itmp1+24);
                _va4 = _mm256_loadu_ps(itmp1+32);

                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);
                _va3 = _mm256_loadu_ps(itmp2+24);
                _va4 = _mm256_loadu_ps(itmp2+32);

                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);
                _sum1 = _mm256_fmadd_ps(_va2, _vb0, _sum1);
                _sum1 = _mm256_fmadd_ps(_va3, _vb1, _sum1);
                _sum1 = _mm256_fmadd_ps(_va4, _vb2, _sum1);

                _mm256_storeu_ps(otmp, _sum0);
                _mm256_storeu_ps(otmp+8, _sum1);

                itmp0 += 32;
                itmp1 += 32;
                itmp2 += 32;
                otmp  += 16;
            }

            for(;j < outw;j++)
            {
                __m256 _sum0 = _mm256_loadu_ps(btmp);

                __m256 _va0 = _mm256_loadu_ps(itmp0);
                __m256 _va1 = _mm256_loadu_ps(itmp0+8);
                __m256 _va2 = _mm256_loadu_ps(itmp0+16);

                __m256 _vb0 = _mm256_loadu_ps(ktmp);
                __m256 _vb1 = _mm256_loadu_ps(ktmp+8);
                __m256 _vb2 = _mm256_loadu_ps(ktmp+16);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _va0 = _mm256_loadu_ps(itmp1);
                _va1 = _mm256_loadu_ps(itmp1+8);
                _va2 = _mm256_loadu_ps(itmp1+16);

                _vb0 = _mm256_loadu_ps(ktmp+24);
                _vb1 = _mm256_loadu_ps(ktmp+32);
                _vb2 = _mm256_loadu_ps(ktmp+40);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _va0 = _mm256_loadu_ps(itmp2);
                _va1 = _mm256_loadu_ps(itmp2+8);
                _va2 = _mm256_loadu_ps(itmp2+16);

                _vb0 = _mm256_loadu_ps(ktmp+48);
                _vb1 = _mm256_loadu_ps(ktmp+56);
                _vb2 = _mm256_loadu_ps(ktmp+64);

                _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0);
                _sum0 = _mm256_fmadd_ps(_va1, _vb1, _sum0);
                _sum0 = _mm256_fmadd_ps(_va2, _vb2, _sum0);

                _mm256_storeu_ps(otmp, _sum0);

                itmp0 += 16;
                itmp1 += 16;
                itmp2 += 16;
                otmp  += 8;
            }
        }
    }
    //load_data
    {
        for(int i = 0; i < channel_count; i++)
        {
            float* otmp = output_tmp + i * 8 * outwh;

            float* tmp0 = output + i * 8 * outwh;
            float* tmp1 = output + i * 8 * outwh + 1 * outwh;
            float* tmp2 = output + i * 8 * outwh + 2 * outwh;
            float* tmp3 = output + i * 8 * outwh + 3 * outwh;
            float* tmp4 = output + i * 8 * outwh + 4 * outwh;
            float* tmp5 = output + i * 8 * outwh + 5 * outwh;
            float* tmp6 = output + i * 8 * outwh + 6 * outwh;
            float* tmp7 = output + i * 8 * outwh + 7 * outwh;
            for(int i = 0; i < outwh; i++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                tmp4[0] = otmp[4];
                tmp5[0] = otmp[5];
                tmp6[0] = otmp[6];
                tmp7[0] = otmp[7];
                otmp += 8;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
                tmp4++;
                tmp5++;
                tmp6++;
                tmp7++;
            }
        }
        int i = 0;
        for(; i + 3 < channel_remain; i += 4)
        {
            int ii = channel_count * 8 + i;
            float* otmp = output_tmp + ii * outwh;

            float* tmp0 = output + ii * outwh;
            float* tmp1 = output + ii * outwh + 1 * outwh;
            float* tmp2 = output + ii * outwh + 2 * outwh;
            float* tmp3 = output + ii * outwh + 3 * outwh;
            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                
                otmp += 8;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
            }
        }

        for(; i < channel_remain; i++)
        {
            int ii = channel_count * 8 + i;
            float* otmp = output_tmp + channel_count * 8 * outwh;

            float* tmp0 = output + ii * outwh;

            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[i];

                otmp += 8;
                tmp0++;
            }
        }
    }
    free(output_tmp);
    free(img_tmp);
    free(kernel_tmp);
    free(bias_tmp);
}
#else//SSE2
static void dwconv3x3s1d1(int inc, int inw, int inh, int outw, int outh, float* kernel_data, float* const img_data, float* const bias_data, float* const output, bool have_biases)
{

    //printf("sse conv3x3s1d1.\n");

    int inwh = inw * inh;
    int outwh = outw * outh;
    int channel_count = inc >> 2;
    int channel_remain = inc - (channel_count << 2);

    //printf("%d %d.\n", channel_count, channel_remain);

    //generate the image tmp
    float* img_tmp = (float*) malloc(4 * inwh * (channel_count + 1) * sizeof(float)); 
    float* kernel_tmp = (float*) malloc(4 * 9 * (channel_count + 1) * sizeof(float));
    float* bias_tmp = (float*) malloc(4 * (channel_count + 1) * sizeof(float));
    {
        for(int i = 0; i < channel_count; i++)
        {
            int ii = i * 4;
            float* k0 = img_data + (ii + 0) * inwh;
            float* k1 = img_data + (ii + 1) * inwh;
            float* k2 = img_data + (ii + 2) * inwh;
            float* k3 = img_data + (ii + 3) * inwh;
        
            float* f0 = kernel_data + (ii + 0) * 9;
            float* f1 = kernel_data + (ii + 1) * 9;
            float* f2 = kernel_data + (ii + 2) * 9;
            float* f3 = kernel_data + (ii + 3) * 9;

            float* b0 = bias_data + (ii + 0);
            float* b1 = bias_data + (ii + 1);
            float* b2 = bias_data + (ii + 2);
            float* b3 = bias_data + (ii + 3);

            float* tmp0 = img_tmp + ii * inwh;
            float* tmp1 = kernel_tmp + ii * 9;
            float* tmp2 = bias_tmp + ii;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
            
                tmp0 += 4;

                k0++;
                k1++;
                k2++;
                k3++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                
                tmp1 += 4;
                f0++;
                f1++;
                f2++;
                f3++;
            }
            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
            }
        }

        for(int i = 0; i < channel_remain; i++)
        {
            int ii = channel_count * 4 + i;
            float* k0 = img_data + ii * inwh;
            float* f0 = kernel_data + ii * 9;
            float* b0 = bias_data + ii; 

            float* tmp0 = img_tmp + channel_count * 4 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 4 * 9;
            float* tmp2 = bias_tmp + channel_count * 4;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[i] = k0[0];

                tmp0 += 4;
                k0++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[i] = f0[0];
                                
                tmp1 += 4;
                f0++;
            }
            if(have_biases)
            {
                tmp2[i] = b0[0];
            }
            else
            {
                tmp2[i] = 0;
            }
        }
    }
    float* output_tmp = (float*) malloc(outwh * 4 * (channel_count + 1) * sizeof(float));
    for(int c = 0; c < channel_count + 1; c++)
    {
        float* ktmp = kernel_tmp + c * 4 * 9;
        float* btmp = bias_tmp + c * 4;
        for(int i = 0; i < outh; i++)
        {
            int j = 0;

            float* itmp0 = img_tmp + c * 4 * inwh + 4 * i * inw;
            float* itmp1 = img_tmp + c * 4 * inwh + 4 * (i + 1) * inw;
            float* itmp2 = img_tmp + c * 4 * inwh + 4 * (i + 2) * inw;
            float* otmp = output_tmp + c * 4 * outwh+ 4 * i * outw;
            for(; j + 7 < outw; j += 8)
            {
#if __SSE__
                __m128 _sum0 = _mm_loadu_ps(btmp);
                __m128 _sum1 = _mm_loadu_ps(btmp);
                __m128 _sum2 = _mm_loadu_ps(btmp);
                __m128 _sum3 = _mm_loadu_ps(btmp);
                __m128 _sum4 = _mm_loadu_ps(btmp);
                __m128 _sum5 = _mm_loadu_ps(btmp);
                __m128 _sum6 = _mm_loadu_ps(btmp);
                __m128 _sum7 = _mm_loadu_ps(btmp);

                __m128 _va0 = _mm_loadu_ps(itmp0);
                __m128 _va1 = _mm_loadu_ps(itmp0+4);
                __m128 _va2 = _mm_loadu_ps(itmp0+8);
                __m128 _va3 = _mm_loadu_ps(itmp0+12);
                __m128 _va4 = _mm_loadu_ps(itmp0+16);
                __m128 _va5 = _mm_loadu_ps(itmp0+20);
                __m128 _va6 = _mm_loadu_ps(itmp0+24);
                __m128 _va7 = _mm_loadu_ps(itmp0+28);
                __m128 _va8 = _mm_loadu_ps(itmp0+32);
                __m128 _va9 = _mm_loadu_ps(itmp0+36);
                
                __m128 _vb0 = _mm_loadu_ps(ktmp);
                __m128 _vb1 = _mm_loadu_ps(ktmp+4);
                __m128 _vb2 = _mm_loadu_ps(ktmp+8);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va4, _vb0), _sum4);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va5, _vb0), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va5, _vb1), _sum4);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va6, _vb1), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va6, _vb2), _sum4);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va6, _vb0), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va7, _vb0), _sum7);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va7, _vb2), _sum5);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va7, _vb1), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va8, _vb1), _sum7);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va8, _vb2), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va9, _vb2), _sum7);

                _va0 = _mm_loadu_ps(itmp1);
                _va1 = _mm_loadu_ps(itmp1+4);
                _va2 = _mm_loadu_ps(itmp1+8);
                _va3 = _mm_loadu_ps(itmp1+12);
                _va4 = _mm_loadu_ps(itmp1+16);
                _va5 = _mm_loadu_ps(itmp1+20);
                _va6 = _mm_loadu_ps(itmp1+24);
                _va7 = _mm_loadu_ps(itmp1+28);
                _va8 = _mm_loadu_ps(itmp1+32);
                _va9 = _mm_loadu_ps(itmp1+36);
                
                _vb0 = _mm_loadu_ps(ktmp+12);
                _vb1 = _mm_loadu_ps(ktmp+16);
                _vb2 = _mm_loadu_ps(ktmp+20);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va4, _vb0), _sum4);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va5, _vb0), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va5, _vb1), _sum4);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va6, _vb1), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va6, _vb2), _sum4);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va6, _vb0), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va7, _vb0), _sum7);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va7, _vb2), _sum5);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va7, _vb1), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va8, _vb1), _sum7);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va8, _vb2), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va9, _vb2), _sum7);

                _va0 = _mm_loadu_ps(itmp2);
                _va1 = _mm_loadu_ps(itmp2+4);
                _va2 = _mm_loadu_ps(itmp2+8);
                _va3 = _mm_loadu_ps(itmp2+12);
                _va4 = _mm_loadu_ps(itmp2+16);
                _va5 = _mm_loadu_ps(itmp2+20);
                _va6 = _mm_loadu_ps(itmp2+24);
                _va7 = _mm_loadu_ps(itmp2+28);
                _va8 = _mm_loadu_ps(itmp2+32);
                _va9 = _mm_loadu_ps(itmp2+36);
                
                _vb0 = _mm_loadu_ps(ktmp+24);
                _vb1 = _mm_loadu_ps(ktmp+28);
                _vb2 = _mm_loadu_ps(ktmp+32);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va4, _vb0), _sum4);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va5, _vb0), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va5, _vb1), _sum4);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va6, _vb1), _sum5);
                _sum4 = _mm_add_ps(_mm_mul_ps(_va6, _vb2), _sum4);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va6, _vb0), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va7, _vb0), _sum7);
                _sum5 = _mm_add_ps(_mm_mul_ps(_va7, _vb2), _sum5);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va7, _vb1), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va8, _vb1), _sum7);
                _sum6 = _mm_add_ps(_mm_mul_ps(_va8, _vb2), _sum6);
                _sum7 = _mm_add_ps(_mm_mul_ps(_va9, _vb2), _sum7);
            
                _mm_storeu_ps(otmp, _sum0);
                _mm_storeu_ps(otmp+4, _sum1);
                _mm_storeu_ps(otmp+8, _sum2);
                _mm_storeu_ps(otmp+12, _sum3);
                _mm_storeu_ps(otmp+16, _sum4);
                _mm_storeu_ps(otmp+20, _sum5);
                _mm_storeu_ps(otmp+24, _sum6);
                _mm_storeu_ps(otmp+28, _sum7);

#else
                float sum0[4] = {btmp[0]};
                float sum1[4] = {btmp[0]};
                float sum2[4] = {btmp[0]};
                float sum3[4] = {btmp[0]};
                float sum4[4] = {btmp[0]};
                float sum5[4] = {btmp[0]};
                float sum6[4] = {btmp[0]};
                float sum7[4] = {btmp[0]};
                for(int k = 0; k < 4; k++)
                {
                    sum0[k] += itmp0[k] * ktmp[k];
                    sum0[k] += itmp1[k] * ktmp[k+12];
                    sum0[k] += itmp2[k] * ktmp[k+24];
                    sum0[k] += itmp0[k+4] * ktmp[k+4];
                    sum0[k] += itmp1[k+4] * ktmp[k+16];
                    sum0[k] += itmp2[k+4] * ktmp[k+28];
                    sum0[k] += itmp0[k+8] * ktmp[k+8];
                    sum0[k] += itmp1[k+8] * ktmp[k+20];
                    sum0[k] += itmp2[k+8] * ktmp[k+32];
                    
                    sum1[k] += itmp0[k+4] * ktmp[k];
                    sum1[k] += itmp1[k+4] * ktmp[k+12];
                    sum1[k] += itmp2[k+4] * ktmp[k+24];
                    sum1[k] += itmp0[k+8] * ktmp[k+4];
                    sum1[k] += itmp1[k+8] * ktmp[k+16];
                    sum1[k] += itmp2[k+8] * ktmp[k+28];
                    sum1[k] += itmp0[k+12] * ktmp[k+8];
                    sum1[k] += itmp1[k+12] * ktmp[k+20];
                    sum1[k] += itmp2[k+12] * ktmp[k+32];
                    
                    sum2[k] += itmp0[k+8] * ktmp[k];
                    sum2[k] += itmp1[k+8] * ktmp[k+12];
                    sum2[k] += itmp2[k+8] * ktmp[k+24];
                    sum2[k] += itmp0[k+12] * ktmp[k+4];
                    sum2[k] += itmp1[k+12] * ktmp[k+16];
                    sum2[k] += itmp2[k+12] * ktmp[k+28];
                    sum2[k] += itmp0[k+16] * ktmp[k+8];
                    sum2[k] += itmp1[k+16] * ktmp[k+20];
                    sum2[k] += itmp2[k+16] * ktmp[k+32];
                    
                    sum3[k] += itmp0[k+12] * ktmp[k];
                    sum3[k] += itmp1[k+12] * ktmp[k+12];
                    sum3[k] += itmp2[k+12] * ktmp[k+24];
                    sum3[k] += itmp0[k+16] * ktmp[k+4];
                    sum3[k] += itmp1[k+16] * ktmp[k+16];
                    sum3[k] += itmp2[k+16] * ktmp[k+28];
                    sum3[k] += itmp0[k+20] * ktmp[k+8];
                    sum3[k] += itmp1[k+20] * ktmp[k+20];
                    sum3[k] += itmp2[k+20] * ktmp[k+32];
                    
                    sum4[k] += itmp0[k+16] * ktmp[k];
                    sum4[k] += itmp1[k+16] * ktmp[k+12];
                    sum4[k] += itmp2[k+16] * ktmp[k+24];
                    sum4[k] += itmp0[k+20] * ktmp[k+4];
                    sum4[k] += itmp1[k+20] * ktmp[k+16];
                    sum4[k] += itmp2[k+20] * ktmp[k+28];
                    sum4[k] += itmp0[k+24] * ktmp[k+8];
                    sum4[k] += itmp1[k+24] * ktmp[k+20];
                    sum4[k] += itmp2[k+24] * ktmp[k+32];

                    sum5[k] += itmp0[k+20] * ktmp[k];
                    sum5[k] += itmp1[k+20] * ktmp[k+12];
                    sum5[k] += itmp2[k+20] * ktmp[k+24];
                    sum5[k] += itmp0[k+24] * ktmp[k+4];
                    sum5[k] += itmp1[k+24] * ktmp[k+16];
                    sum5[k] += itmp2[k+24] * ktmp[k+28];
                    sum5[k] += itmp0[k+28] * ktmp[k+8];
                    sum5[k] += itmp1[k+28] * ktmp[k+20];
                    sum5[k] += itmp2[k+28] * ktmp[k+32];

                    sum6[k] += itmp0[k+24] * ktmp[k];
                    sum6[k] += itmp1[k+24] * ktmp[k+12];
                    sum6[k] += itmp2[k+24] * ktmp[k+24];
                    sum6[k] += itmp0[k+28] * ktmp[k+4];
                    sum6[k] += itmp1[k+28] * ktmp[k+16];
                    sum6[k] += itmp2[k+28] * ktmp[k+28];
                    sum6[k] += itmp0[k+32] * ktmp[k+8];
                    sum6[k] += itmp1[k+32] * ktmp[k+20];
                    sum6[k] += itmp2[k+32] * ktmp[k+32];

                    sum7[k] += itmp0[k+28] * ktmp[k];
                    sum7[k] += itmp1[k+28] * ktmp[k+12];
                    sum7[k] += itmp2[k+28] * ktmp[k+24];
                    sum7[k] += itmp0[k+32] * ktmp[k+4];
                    sum7[k] += itmp1[k+32] * ktmp[k+16];
                    sum7[k] += itmp2[k+32] * ktmp[k+28];
                    sum7[k] += itmp0[k+36] * ktmp[k+8];
                    sum7[k] += itmp1[k+36] * ktmp[k+20];
                    sum7[k] += itmp2[k+36] * ktmp[k+32];
                }

                for(int k = 0; k < 4; k++)
                {
                    otmp[k]    = sum0[k];
                    otmp[k+4]  = sum1[k];
                    otmp[k+8]  = sum2[k];
                    otmp[k+12] = sum3[k];
                    otmp[k+16] = sum4[k];
                    otmp[k+20] = sum5[k];
                    otmp[k+24] = sum6[k];
                    otmp[k+28] = sum7[k];
                }
#endif
                itmp0 += 32;
                itmp1 += 32;
                itmp2 += 32;
                otmp  += 32;
            }

            for(; j + 3 < outw; j += 4)
            {
#if __SSE__
                __m128 _sum0 = _mm_loadu_ps(btmp);
                __m128 _sum1 = _mm_loadu_ps(btmp);
                __m128 _sum2 = _mm_loadu_ps(btmp);
                __m128 _sum3 = _mm_loadu_ps(btmp);

                __m128 _va0 = _mm_loadu_ps(itmp0);
                __m128 _va1 = _mm_loadu_ps(itmp0+4);
                __m128 _va2 = _mm_loadu_ps(itmp0+8);
                __m128 _va3 = _mm_loadu_ps(itmp0+12);
                __m128 _va4 = _mm_loadu_ps(itmp0+16);
                __m128 _va5 = _mm_loadu_ps(itmp0+20);
                
                __m128 _vb0 = _mm_loadu_ps(ktmp);
                __m128 _vb1 = _mm_loadu_ps(ktmp+4);
                __m128 _vb2 = _mm_loadu_ps(ktmp+8);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);

                _va0 = _mm_loadu_ps(itmp1);
                _va1 = _mm_loadu_ps(itmp1+4);
                _va2 = _mm_loadu_ps(itmp1+8);
                _va3 = _mm_loadu_ps(itmp1+12);
                _va4 = _mm_loadu_ps(itmp1+16);
                _va5 = _mm_loadu_ps(itmp1+20);
                
                _vb0 = _mm_loadu_ps(ktmp+12);
                _vb1 = _mm_loadu_ps(ktmp+16);
                _vb2 = _mm_loadu_ps(ktmp+20);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);

                _va0 = _mm_loadu_ps(itmp2);
                _va1 = _mm_loadu_ps(itmp2+4);
                _va2 = _mm_loadu_ps(itmp2+8);
                _va3 = _mm_loadu_ps(itmp2+12);
                _va4 = _mm_loadu_ps(itmp2+16);
                _va5 = _mm_loadu_ps(itmp2+20);
                
                _vb0 = _mm_loadu_ps(ktmp+24);
                _vb1 = _mm_loadu_ps(ktmp+28);
                _vb2 = _mm_loadu_ps(ktmp+32);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va1, _vb0), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va2, _vb1), _sum1);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va2, _vb0), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va3, _vb0), _sum3);
                _sum1 = _mm_add_ps(_mm_mul_ps(_va3, _vb2), _sum1);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va3, _vb1), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va4, _vb1), _sum3);
                _sum2 = _mm_add_ps(_mm_mul_ps(_va4, _vb2), _sum2);
                _sum3 = _mm_add_ps(_mm_mul_ps(_va5, _vb2), _sum3);
            
                _mm_storeu_ps(otmp, _sum0);
                _mm_storeu_ps(otmp+4, _sum1);
                _mm_storeu_ps(otmp+8, _sum2);
                _mm_storeu_ps(otmp+12, _sum3);

#else
                float sum0[4] = {btmp[0]};
                float sum1[4] = {btmp[0]};
                float sum2[4] = {btmp[0]};
                float sum3[4] = {btmp[0]};
                for(int k = 0; k < 4; k++)
                {
                    sum0[k] += itmp0[k] * ktmp[k];
                    sum0[k] += itmp1[k] * ktmp[k+12];
                    sum0[k] += itmp2[k] * ktmp[k+24];
                    sum0[k] += itmp0[k+4] * ktmp[k+4];
                    sum0[k] += itmp1[k+4] * ktmp[k+16];
                    sum0[k] += itmp2[k+4] * ktmp[k+28];
                    sum0[k] += itmp0[k+8] * ktmp[k+8];
                    sum0[k] += itmp1[k+8] * ktmp[k+20];
                    sum0[k] += itmp2[k+8] * ktmp[k+32];
                    
                    sum1[k] += itmp0[k+4] * ktmp[k];
                    sum1[k] += itmp1[k+4] * ktmp[k+12];
                    sum1[k] += itmp2[k+4] * ktmp[k+24];
                    sum1[k] += itmp0[k+8] * ktmp[k+4];
                    sum1[k] += itmp1[k+8] * ktmp[k+16];
                    sum1[k] += itmp2[k+8] * ktmp[k+28];
                    sum1[k] += itmp0[k+12] * ktmp[k+8];
                    sum1[k] += itmp1[k+12] * ktmp[k+20];
                    sum1[k] += itmp2[k+12] * ktmp[k+32];
                    
                    sum2[k] += itmp0[k+8] * ktmp[k];
                    sum2[k] += itmp1[k+8] * ktmp[k+12];
                    sum2[k] += itmp2[k+8] * ktmp[k+24];
                    sum2[k] += itmp0[k+12] * ktmp[k+4];
                    sum2[k] += itmp1[k+12] * ktmp[k+16];
                    sum2[k] += itmp2[k+12] * ktmp[k+28];
                    sum2[k] += itmp0[k+16] * ktmp[k+8];
                    sum2[k] += itmp1[k+16] * ktmp[k+20];
                    sum2[k] += itmp2[k+16] * ktmp[k+32];
                    
                    sum3[k] += itmp0[k+12] * ktmp[k];
                    sum3[k] += itmp1[k+12] * ktmp[k+12];
                    sum3[k] += itmp2[k+12] * ktmp[k+24];
                    sum3[k] += itmp0[k+16] * ktmp[k+4];
                    sum3[k] += itmp1[k+16] * ktmp[k+16];
                    sum3[k] += itmp2[k+16] * ktmp[k+28];
                    sum3[k] += itmp0[k+20] * ktmp[k+8];
                    sum3[k] += itmp1[k+20] * ktmp[k+20];
                    sum3[k] += itmp2[k+20] * ktmp[k+32];
                }
                for(int k = 0; k < 4; k++)
                {
                    otmp[k]    = sum0[k];
                    otmp[k+4]  = sum1[k];
                    otmp[k+8]  = sum2[k];
                    otmp[k+12] = sum3[k];
                }
#endif
                itmp0 += 16;
                itmp1 += 16;
                itmp2 += 16;
                otmp  += 16;
            }

            for(;j < outw;j++)
            {
#if __SSE__
                __m128 _sum0 = _mm_loadu_ps(btmp);

                __m128 _va0 = _mm_loadu_ps(itmp0);
                __m128 _va1 = _mm_loadu_ps(itmp0+4);
                __m128 _va2 = _mm_loadu_ps(itmp0+8);

                __m128 _vb0 = _mm_loadu_ps(ktmp);
                __m128 _vb1 = _mm_loadu_ps(ktmp+4);
                __m128 _vb2 = _mm_loadu_ps(ktmp+8);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);

                _va0 = _mm_loadu_ps(itmp1);
                _va1 = _mm_loadu_ps(itmp1+4);
                _va2 = _mm_loadu_ps(itmp1+8);

                _vb0 = _mm_loadu_ps(ktmp+12);
                _vb1 = _mm_loadu_ps(ktmp+16);
                _vb2 = _mm_loadu_ps(ktmp+20);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);

                _va0 = _mm_loadu_ps(itmp2);
                _va1 = _mm_loadu_ps(itmp2+4);
                _va2 = _mm_loadu_ps(itmp2+8);

                _vb0 = _mm_loadu_ps(ktmp+24);
                _vb1 = _mm_loadu_ps(ktmp+28);
                _vb2 = _mm_loadu_ps(ktmp+32);

                _sum0 = _mm_add_ps(_mm_mul_ps(_va0, _vb0), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va1, _vb1), _sum0);
                _sum0 = _mm_add_ps(_mm_mul_ps(_va2, _vb2), _sum0);

                _mm_storeu_ps(otmp, _sum0);
#else
                float sum0[4] = {btmp[0]};
                for(int k = 0; k < 4; k++)
                {
                    sum0[k] += itmp0[k] * ktmp[k];
                    sum0[k] += itmp1[k] * ktmp[k+12];
                    sum0[k] += itmp2[k] * ktmp[k+24];
                    sum0[k] += itmp0[k+4] * ktmp[k+4];
                    sum0[k] += itmp1[k+4] * ktmp[k+16];
                    sum0[k] += itmp2[k+4] * ktmp[k+28];
                    sum0[k] += itmp0[k+8] * ktmp[k+8];
                    sum0[k] += itmp1[k+8] * ktmp[k+20];
                    sum0[k] += itmp2[k+8] * ktmp[k+32];
                }

                for(int k = 0; k < 4; k++)
                {
                    otmp[k] = sum0[k];
                }
#endif
                itmp0 += 4;
                itmp1 += 4;
                itmp2 += 4;
                otmp  += 4;
            }
        }
    }
    
    {
        for(int i = 0; i < channel_count; i++)
        {
            float* otmp = output_tmp + i * 4 * outwh;

            float* tmp0 = output + i * 4 * outwh;
            float* tmp1 = output + i * 4 * outwh + 1 * outwh;
            float* tmp2 = output + i * 4 * outwh + 2 * outwh;
            float* tmp3 = output + i * 4 * outwh + 3 * outwh;
            for(int i = 0; i < outwh; i++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                otmp += 4;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
            }
        }

        for(int i = 0; i < channel_remain; i++)
        {
            int ii = channel_count * 4 + i;
            float* otmp = output_tmp + channel_count * 4 * outwh;

            float* tmp0 = output + ii * outwh;

            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[i];

                otmp += 4;
                tmp0++;
            }
        }
    }

    free(output_tmp);
    free(img_tmp);
    free(kernel_tmp);
    free(bias_tmp);
}

static void dwconv3x3s2d1(int inc, int inw, int inh, int outw, int outh, float* kernel_data, float* const img_data, float* const bias_data, float* const output, bool have_biases)
{
    int inwh = inw * inh;
    int outwh = outw * outh;
    int channel_count = inc >> 2;
    int channel_remain = inc - (channel_count << 2);
    //generate the image tmp
    float* img_tmp = (float*) malloc(4 * inwh * (channel_count + 1) * sizeof(float)); 
    float* kernel_tmp = (float*) malloc(4 * 9 * (channel_count + 1) * sizeof(float));
    float* bias_tmp = (float*) malloc(4 * (channel_count + 1) * sizeof(float));
    {
        for(int i = 0; i < channel_count; i++)
        {
            int ii = i * 4;
            float* k0 = img_data + (ii + 0) * inwh;
            float* k1 = img_data + (ii + 1) * inwh;
            float* k2 = img_data + (ii + 2) * inwh;
            float* k3 = img_data + (ii + 3) * inwh;
        
            float* f0 = kernel_data + (ii + 0) * 9;
            float* f1 = kernel_data + (ii + 1) * 9;
            float* f2 = kernel_data + (ii + 2) * 9;
            float* f3 = kernel_data + (ii + 3) * 9;

            float* b0 = bias_data + (ii + 0);
            float* b1 = bias_data + (ii + 1);
            float* b2 = bias_data + (ii + 2);
            float* b3 = bias_data + (ii + 3);

            float* tmp0 = img_tmp + ii * inwh;
            float* tmp1 = kernel_tmp + ii * 9;
            float* tmp2 = bias_tmp + ii;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[0] = k0[0];
                tmp0[1] = k1[0];
                tmp0[2] = k2[0];
                tmp0[3] = k3[0];
            
                tmp0 += 4;

                k0++;
                k1++;
                k2++;
                k3++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[0] = f0[0];
                tmp1[1] = f1[0];
                tmp1[2] = f2[0];
                tmp1[3] = f3[0];
                
                tmp1 += 4;
                f0++;
                f1++;
                f2++;
                f3++;
            }
            if(have_biases)
            {
                tmp2[0] = b0[0];
                tmp2[1] = b1[0];
                tmp2[2] = b2[0];
                tmp2[3] = b3[0];
            }
            else
            {
                tmp2[0] = 0;
                tmp2[1] = 0;
                tmp2[2] = 0;
                tmp2[3] = 0;
            }
        }

        for(int i = 0; i < channel_remain; i++)
        {
            int ii = channel_count * 4 + i;
            float* k0 = img_data + ii * inwh;
            float* f0 = kernel_data + ii * 9;
            float* b0 = bias_data + ii; 

            float* tmp0 = img_tmp + channel_count * 4 * inwh;
            float* tmp1 = kernel_tmp + channel_count * 4 * 9;
            float* tmp2 = bias_tmp + channel_count * 4;

            for(int j = 0; j < inwh; j++)
            {
                tmp0[i] = k0[0];

                tmp0 += 4;
                k0++;
            }

            for(int j = 0; j < 9; j++)
            {
                tmp1[i] = f0[0];
                                
                tmp1 += 4;
                f0++;
            }
            if(have_biases)
            {
                tmp2[i] = b0[0];
            }
            else
            {
                tmp2[i] = 0;
            }
        }
    }
    float* output_tmp = (float*) malloc(outwh * 4 * (channel_count + 1) * sizeof(float));
    for(int c = 0; c < channel_count + 1; c++)
    {
        float* ktmp = kernel_tmp + c * 4 * 9;
        float* btmp = bias_tmp + c * 4;
        for(int i = 0; i < outh; i++)
        {
            int j = 0;

            float* itmp0 = img_tmp + c * 4 * inwh + 4 * i * 2 * inw;
            float* itmp1 = img_tmp + c * 4 * inwh + 4 * (i * 2 + 1) * inw;
            float* itmp2 = img_tmp + c * 4 * inwh + 4 * (i * 2 + 2) * inw;
            float* otmp = output_tmp + c * 4 * outwh+ 4 * i * outw;
            for(; j + 3 < outw; j += 4)
            {
#if __SSE__
                __m128 _sum0 = _mm_loadu_ps(btmp);
                __m128 _sum1 = _mm_loadu_ps(btmp);
                __m128 _sum2 = _mm_loadu_ps(btmp);
                __m128 _sum3 = _mm_loadu_ps(btmp);

                __m128 _va0 = _mm_loadu_ps(itmp0);
                __m128 _va1 = _mm_loadu_ps(itmp0+4);
                __m128 _va2 = _mm_loadu_ps(itmp0+8);
                __m128 _va3 = _mm_loadu_ps(itmp0+12);
                __m128 _va4 = _mm_loadu_ps(itmp0+16);
                __m128 _va5 = _mm_loadu_ps(itmp0+20);
                __m128 _va6 = _mm_loadu_ps(itmp0+24);
                __m128 _va7 = _mm_loadu_ps(itmp0+28);
                __m128 _va8 = _mm_loadu_ps(itmp0+32);
                
                __m128 _vb0 = _mm_loadu_ps(ktmp);
                __m128 _vb1 = _mm_loadu_ps(ktmp+4);
                __m128 _vb2 = _mm_loadu_ps(ktmp+8);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va2, _vb0));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va3, _vb1));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va4, _vb2));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va4, _vb0));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va5, _vb1));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va6, _vb2));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va6, _vb0));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va7, _vb1));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va8, _vb2));

                _va0 = _mm_loadu_ps(itmp1);
                _va1 = _mm_loadu_ps(itmp1+4);
                _va2 = _mm_loadu_ps(itmp1+8);
                _va3 = _mm_loadu_ps(itmp1+12);
                _va4 = _mm_loadu_ps(itmp1+16);
                _va5 = _mm_loadu_ps(itmp1+20);
                _va6 = _mm_loadu_ps(itmp1+24);
                _va7 = _mm_loadu_ps(itmp1+28);
                _va8 = _mm_loadu_ps(itmp1+32);
                
                _vb0 = _mm_loadu_ps(ktmp+12);
                _vb1 = _mm_loadu_ps(ktmp+16);
                _vb2 = _mm_loadu_ps(ktmp+20);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va2, _vb0));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va3, _vb1));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va4, _vb2));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va4, _vb0));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va5, _vb1));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va6, _vb2));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va6, _vb0));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va7, _vb1));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va8, _vb2));

                _va0 = _mm_loadu_ps(itmp2);
                _va1 = _mm_loadu_ps(itmp2+4);
                _va2 = _mm_loadu_ps(itmp2+8);
                _va3 = _mm_loadu_ps(itmp2+12);
                _va4 = _mm_loadu_ps(itmp2+16);
                _va5 = _mm_loadu_ps(itmp2+20);
                _va6 = _mm_loadu_ps(itmp2+24);
                _va7 = _mm_loadu_ps(itmp2+28);
                _va8 = _mm_loadu_ps(itmp2+32);
                
                _vb0 = _mm_loadu_ps(ktmp+24);
                _vb1 = _mm_loadu_ps(ktmp+28);
                _vb2 = _mm_loadu_ps(ktmp+32);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va2, _vb0));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va3, _vb1));
                _sum1 = _mm_add_ps(_sum1, _mm_mul_ps(_va4, _vb2));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va4, _vb0));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va5, _vb1));
                _sum2 = _mm_add_ps(_sum2, _mm_mul_ps(_va6, _vb2));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va6, _vb0));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va7, _vb1));
                _sum3 = _mm_add_ps(_sum3, _mm_mul_ps(_va8, _vb2));
            
                _mm_storeu_ps(otmp, _sum0);
                _mm_storeu_ps(otmp+4, _sum1);
                _mm_storeu_ps(otmp+8, _sum2);
                _mm_storeu_ps(otmp+12, _sum3);

#else
                float sum0[4] = {btmp[0]};
                float sum1[4] = {btmp[0]};
                float sum2[4] = {btmp[0]};
                float sum3[4] = {btmp[0]};
                for(int k = 0; k < 4; k++)
                {
                    sum0[k] += itmp0[k] * ktmp[k];
                    sum0[k] += itmp1[k] * ktmp[k+12];
                    sum0[k] += itmp2[k] * ktmp[k+24];
                    sum0[k] += itmp0[k+4] * ktmp[k+4];
                    sum0[k] += itmp1[k+4] * ktmp[k+16];
                    sum0[k] += itmp2[k+4] * ktmp[k+28];
                    sum0[k] += itmp0[k+8] * ktmp[k+8];
                    sum0[k] += itmp1[k+8] * ktmp[k+20];
                    sum0[k] += itmp2[k+8] * ktmp[k+32];
                    
                    sum1[k] += itmp0[k+8] * ktmp[k];
                    sum1[k] += itmp1[k+8] * ktmp[k+12];
                    sum1[k] += itmp2[k+8] * ktmp[k+24];
                    sum1[k] += itmp0[k+12] * ktmp[k+4];
                    sum1[k] += itmp1[k+12] * ktmp[k+16];
                    sum1[k] += itmp2[k+12] * ktmp[k+28];
                    sum1[k] += itmp0[k+16] * ktmp[k+8];
                    sum1[k] += itmp1[k+16] * ktmp[k+20];
                    sum1[k] += itmp2[k+16] * ktmp[k+32];
                    
                    sum2[k] += itmp0[k+16] * ktmp[k];
                    sum2[k] += itmp1[k+16] * ktmp[k+12];
                    sum2[k] += itmp2[k+16] * ktmp[k+24];
                    sum2[k] += itmp0[k+20] * ktmp[k+4];
                    sum2[k] += itmp1[k+20] * ktmp[k+16];
                    sum2[k] += itmp2[k+20] * ktmp[k+28];
                    sum2[k] += itmp0[k+24] * ktmp[k+8];
                    sum2[k] += itmp1[k+24] * ktmp[k+20];
                    sum2[k] += itmp2[k+24] * ktmp[k+32];
                    
                    sum3[k] += itmp0[k+24] * ktmp[k];
                    sum3[k] += itmp1[k+24] * ktmp[k+12];
                    sum3[k] += itmp2[k+24] * ktmp[k+24];
                    sum3[k] += itmp0[k+28] * ktmp[k+4];
                    sum3[k] += itmp1[k+28] * ktmp[k+16];
                    sum3[k] += itmp2[k+28] * ktmp[k+28];
                    sum3[k] += itmp0[k+32] * ktmp[k+8];
                    sum3[k] += itmp1[k+32] * ktmp[k+20];
                    sum3[k] += itmp2[k+32] * ktmp[k+32];
                }

                for(int k = 0; k < 4; k++)
                {
                    otmp[k]    = sum0[k];
                    otmp[k+4]  = sum1[k];
                    otmp[k+8]  = sum2[k];
                    otmp[k+12] = sum3[k];
                }
#endif
                itmp0 += 32;
                itmp1 += 32;
                itmp2 += 32;
                otmp  += 16;
            }

            for(;j < outw;j++)
            {
#if __SSE__
                __m128 _sum0 = _mm_loadu_ps(btmp);
                __m128 _va0 = _mm_loadu_ps(itmp0);
                __m128 _va1 = _mm_loadu_ps(itmp0+4);
                __m128 _va2 = _mm_loadu_ps(itmp0+8);

                __m128 _vb0 = _mm_loadu_ps(ktmp);
                __m128 _vb1 = _mm_loadu_ps(ktmp+4);
                __m128 _vb2 = _mm_loadu_ps(ktmp+8);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));

                _va0 = _mm_loadu_ps(itmp1);
                _va1 = _mm_loadu_ps(itmp1+4);
                _va2 = _mm_loadu_ps(itmp1+8);

                _vb0 = _mm_loadu_ps(ktmp+12);
                _vb1 = _mm_loadu_ps(ktmp+16);
                _vb2 = _mm_loadu_ps(ktmp+20);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));

                _va0 = _mm_loadu_ps(itmp2);
                _va1 = _mm_loadu_ps(itmp2+4);
                _va2 = _mm_loadu_ps(itmp2+8);

                _vb0 = _mm_loadu_ps(ktmp+24);
                _vb1 = _mm_loadu_ps(ktmp+28);
                _vb2 = _mm_loadu_ps(ktmp+32);

                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va0, _vb0));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va1, _vb1));
                _sum0 = _mm_add_ps(_sum0, _mm_mul_ps(_va2, _vb2));

                _mm_storeu_ps(otmp, _sum0);
#else
                float sum0[4] = {btmp[0]};
                for(int k = 0; k < 4; k++)
                {
                    sum0[k] += itmp0[k] * ktmp[k];
                    sum0[k] += itmp1[k] * ktmp[k+12];
                    sum0[k] += itmp2[k] * ktmp[k+24];
                    sum0[k] += itmp0[k+4] * ktmp[k+4];
                    sum0[k] += itmp1[k+4] * ktmp[k+16];
                    sum0[k] += itmp2[k+4] * ktmp[k+28];
                    sum0[k] += itmp0[k+8] * ktmp[k+8];
                    sum0[k] += itmp1[k+8] * ktmp[k+20];
                    sum0[k] += itmp2[k+8] * ktmp[k+32];
                }

                for(int k = 0; k < 4; k++)
                {
                    otmp[k] = sum0[k];
                }
#endif
                itmp0 += 8;
                itmp1 += 8;
                itmp2 += 8;
                otmp  += 4;
            }
        }
    }
    
    {
        for(int i = 0; i < channel_count; i++)
        {
            float* otmp = output_tmp + i * 4 * outwh;

            float* tmp0 = output + i * 4 * outwh;
            float* tmp1 = output + i * 4 * outwh + 1 * outwh;
            float* tmp2 = output + i * 4 * outwh + 2 * outwh;
            float* tmp3 = output + i * 4 * outwh + 3 * outwh;
            for(int i = 0; i < outwh; i++)
            {
                tmp0[0] = otmp[0];
                tmp1[0] = otmp[1];
                tmp2[0] = otmp[2];
                tmp3[0] = otmp[3];
                otmp += 4;
                tmp0++;
                tmp1++;
                tmp2++;
                tmp3++;
            }
        }

        for(int i = 0; i < channel_remain; i++)
        {
            int ii = channel_count * 4 + i;
            float* otmp = output_tmp + channel_count * 4 * outwh;

            float* tmp0 = output + ii * outwh;

            for(int j = 0; j < outwh; j++)
            {
                tmp0[0] = otmp[i];

                otmp += 4;
                tmp0++;
            }
        }
    }
    free(output_tmp);
    free(img_tmp);
    free(kernel_tmp);
    free(bias_tmp);
}
#endif
