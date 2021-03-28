#include <emmintrin.h>
#include <stdio.h>
#include <assert.h>
#include "pooling_param.h"

#define POOL_GENERIC 0
#define POOL_K2S2 1
#define POOL_K3S2 2
#define POOL_K3S1 3

typedef void (*pooling_kernel_t)(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int,
                                 int, int, int, int, int, int pad_h1, int pad_w1, int);

#define max(a, b) (((a) > (b)) ? (a) : (b))

static void avg_2x2s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }

    int loopw = (inw - 1) >> 1;
    int looph = (inh - 1) >> 1;
    int remain_w = inw - outw * 2;
    __m128 scalar_025 = _mm_set1_ps(0.25f);
    __m128 scalar_05 = _mm_set1_ps(0.5f);
    if (inw % 2 == 0)
    {
        remain_w = 1;
    }
    else
    {
        remain_w = 0;
    }
    const float* line0 = input;
    const float* line1;
    float* out_ptr = output;

    __m128 line00;
    __m128 line01;
    __m128 line10;
    __m128 line11;
    __m128 sum0;
    __m128 sum1;
    __m128 sum;

    line00 = _mm_loadu_ps(line0);
    if (is_caffe == 1)
    {
        line00 = _mm_mul_ps(line00, scalar_025);
    }
    _mm_storeu_ps(out_ptr, line00);
    line0 += 4;
    out_ptr += 4;
    for (int i = 0; i < loopw; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        sum0 = _mm_add_ps(line00, line01);
        if (is_caffe == 0)
        {
            sum0 = _mm_mul_ps(sum0, scalar_05);
        }
        else
        {
            sum0 = _mm_mul_ps(sum0, scalar_025);
        }
        _mm_storeu_ps(out_ptr, sum0);
        out_ptr += 4;
        line0 += 8;
    }
    if (inw % 2 == 0)
    {
        line00 = _mm_loadu_ps(line0);
        if (is_caffe == 1)
        {
            line00 = _mm_mul_ps(line00, scalar_025);
        }
        _mm_storeu_ps(out_ptr, line00);
        out_ptr += 4;
    }
    line0 += remain_w * 4;
    line1 = line0 + inw * 4;

    for (int i = 0; i < looph; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line10 = _mm_loadu_ps(line1);
        sum = _mm_add_ps(line00, line10);
        if (is_caffe == 0)
        {
            sum = _mm_mul_ps(sum, scalar_05);
        }
        else
        {
            sum = _mm_mul_ps(sum, scalar_025);
        }
        _mm_storeu_ps(out_ptr, sum);
        out_ptr += 4;
        line0 += 4;
        line1 += 4;
        for (int i = 0; i < loopw; i++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            sum0 = _mm_add_ps(line00, line01);
            sum1 = _mm_add_ps(line10, line11);
            sum = _mm_add_ps(sum0, sum1);
            sum = _mm_mul_ps(sum, scalar_025);
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
            line0 += 8;
            line1 += 8;
        }
        if (inw % 2 == 0)
        {
            line00 = _mm_loadu_ps(line0);
            line10 = _mm_loadu_ps(line1);
            sum = _mm_add_ps(line00, line10);
            if (is_caffe == 0)
            {
                sum = _mm_mul_ps(sum, scalar_05);
            }
            else
            {
                sum = _mm_mul_ps(sum, scalar_025);
            }
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
        }
        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
    }

    if (inh % 2 == 0)
    {
        line00 = _mm_loadu_ps(line0);
        if (is_caffe == 1)
        {
            line00 = _mm_mul_ps(line00, scalar_025);
        }
        _mm_storeu_ps(out_ptr, line00);
        out_ptr += 4;
        line0 += 4;
        for (int i = 0; i < loopw; i++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            sum = _mm_add_ps(line00, line01);
            if (is_caffe == 0)
            {
                sum0 = _mm_mul_ps(sum0, scalar_05);
            }
            else
            {
                sum0 = _mm_mul_ps(sum0, scalar_025);
            }
            _mm_storeu_ps(out_ptr, sum);
            line0 += 8;
            out_ptr += 4;
        }
        if (inw % 2 == 0)
        {
            line00 = _mm_loadu_ps(line0);
            if (is_caffe == 1)
            {
                line00 = _mm_mul_ps(line00, scalar_025);
            }
            _mm_storeu_ps(out_ptr, line00);
            out_ptr += 4;
        }
    }
}

static void max_2x2s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }

    int loopw = (inw - 1) >> 1;
    int looph = (inh - 1) >> 1;
    int remain_w = inw - outw * 2;
    if (inw % 2 == 0)
    {
        remain_w = 1;
    }
    else
    {
        remain_w = 0;
    }
    const float* line0 = input;
    const float* line1;
    float* out_ptr = output;

    __m128 line00;
    __m128 line01;
    __m128 line10;
    __m128 line11;
    __m128 max0;
    __m128 max1;
    __m128 max;

    line00 = _mm_loadu_ps(line0);
    _mm_storeu_ps(out_ptr, line00);
    line0 += 4;
    out_ptr += 4;
    for (int i = 0; i < loopw; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        max0 = _mm_max_ps(line00, line01);
        _mm_storeu_ps(out_ptr, max0);
        out_ptr += 4;
        line0 += 8;
    }
    if (inw % 2 == 0)
    {
        line00 = _mm_loadu_ps(line0);
        _mm_storeu_ps(out_ptr, line00);
        out_ptr += 4;
    }
    line0 += remain_w * 4;
    line1 = line0 + inw * 4;

    for (int i = 0; i < looph; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line10 = _mm_loadu_ps(line1);
        max = _mm_max_ps(line00, line10);
        _mm_storeu_ps(out_ptr, max);
        out_ptr += 4;
        line0 += 4;
        line1 += 4;
        for (int i = 0; i < loopw; i++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            max0 = _mm_max_ps(line00, line01);
            max1 = _mm_max_ps(line10, line11);
            max = _mm_max_ps(max0, max1);
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
            line0 += 8;
            line1 += 8;
        }
        if (inw % 2 == 0)
        {
            line00 = _mm_loadu_ps(line0);
            line10 = _mm_loadu_ps(line1);
            max = _mm_max_ps(line00, line10);
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
        }
        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
    }

    if (inh % 2 == 0)
    {
        line00 = _mm_loadu_ps(line0);
        _mm_storeu_ps(out_ptr, line00);
        out_ptr += 4;
        line0 += 4;
        for (int i = 0; i < loopw; i++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            max = _mm_max_ps(line00, line01);
            _mm_storeu_ps(out_ptr, max);
            line0 += 8;
            out_ptr += 4;
        }
        if (inw % 2 == 0)
        {
            line00 = _mm_loadu_ps(line0);
            _mm_storeu_ps(out_ptr, line00);
            out_ptr += 4;
        }
    }
}

static void avg_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    const float* line0 = input;
    const float* line1 = input + inw * 4;
    float* out_ptr = output;

    __m128 scalar_025 = _mm_set1_ps(0.25f);
    __m128 scalar_05 = _mm_set1_ps(0.5f);
    __m128 line00;
    __m128 line01;
    __m128 line10;
    __m128 line11;
    __m128 add0;
    __m128 add1;
    __m128 add;
    for (int i = 0; i < outh; i++)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            add0 = _mm_add_ps(line00, line01);
            add1 = _mm_add_ps(line10, line11);
            add = _mm_add_ps(add0, add1);
            add = _mm_mul_ps(add, scalar_025);
            _mm_storeu_ps(out_ptr, add);
            line0 += 8;
            line1 += 8;
            out_ptr += 4;
        }
        if (pad_w1 > 0)
        {
            add = _mm_add_ps(line00, line10);
            add = _mm_mul_ps(add, scalar_05);
            _mm_storeu_ps(out_ptr, add);
        }
        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
    }

    if (pad_h1 > 0)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            add0 = _mm_add_ps(line00, line01);
            add0 = _mm_mul_ps(add0, scalar_05);
            _mm_storeu_ps(out_ptr, add0);
            line0 += 8;
            out_ptr += 4;
        }
        if (pad_w1 > 0)
        {
            line00 = _mm_loadu_ps(line0);
            _mm_storeu_ps(out_ptr, line00);
        }
    }
}

static void max_2x2s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;

    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    int block_w = outw >> 2;
    int remain_w = inw - outw * 2;

    const float* line0 = input;
    const float* line1 = input + inw * 4;
    float* out_ptr = output;

    __m128 line00;
    __m128 line01;
    __m128 line10;
    __m128 line11;
    __m128 max0;
    __m128 max1;
    __m128 max;
    for (int i = 0; i < outh; i++)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            max0 = _mm_max_ps(line00, line01);
            max1 = _mm_max_ps(line10, line11);
            max = _mm_max_ps(max0, max1);
            _mm_storeu_ps(out_ptr, max);
            line0 += 8;
            line1 += 8;
            out_ptr += 4;
        }
        if (pad_w1 > 0)
        {
            max = _mm_max_ps(line00, line10);
            _mm_storeu_ps(out_ptr, max);
        }
        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
    }

    if (pad_h1 > 0)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            max0 = _mm_max_ps(line00, line01);
            _mm_storeu_ps(out_ptr, max0);
            line0 += 8;
            out_ptr += 4;
        }
        if (pad_w1 > 0)
        {
            line00 = _mm_loadu_ps(line0);
            _mm_storeu_ps(out_ptr, line00);
        }
    }
}

static void max_3x3s1_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int mid_h = inh - 2;
    int mid_w = inw - 2;

    const float* line1 = input;
    const float* line2 = input + inw * 4;
    float* out_ptr = output;
    __m128 line10 = _mm_loadu_ps(line1);
    __m128 line11 = _mm_loadu_ps(line1 + 4);
    __m128 line20 = _mm_loadu_ps(line2);
    __m128 line21 = _mm_loadu_ps(line2 + 4);

    __m128 max1 = _mm_max_ps(line10, line20);
    __m128 max2 = _mm_max_ps(line11, line21);
    __m128 max12 = _mm_max_ps(max1, max2);
    _mm_storeu_ps(out_ptr, max12);
    out_ptr += 4;

    // h begin center----[line1+=1]----------------------------------
    // for (int j = 0; j < mid_w; j++)
    //  {
    //         float max1 = arm64_max(arm64_max(line1[0], line1[1]), line1[2]);
    //         float max2 = arm64_max(arm64_max(line2[0], line2[1]), line2[2]);
    //         *out_ptr = arm64_max(max2, max1);
    //         out_ptr++;
    //         line1 += 1;
    //         line2 += 1;
    //  }
    __m128 line12;
    __m128 line22;
    __m128 max;
    for (int j = 0; j < mid_w; j++)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        max12 = _mm_max_ps(_mm_max_ps(line10, line20), _mm_max_ps(line11, line21));
        line12 = _mm_loadu_ps(line1 + 8);
        line22 = _mm_loadu_ps(line2 + 8);
        max = _mm_max_ps(line12, line22);
        _mm_storeu_ps(out_ptr, _mm_max_ps(max12, max));
        out_ptr += 4;
        line1 += 4;
        line2 += 4;
    }
    // h begin right----[line1+=2]-----------------------------------
    // *out_ptr = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line2[0], line2[1]));
    // out_ptr++;
    // line1 += 2;
    // line2 += 2;

    line10 = _mm_loadu_ps(line1);
    line11 = _mm_loadu_ps(line1 + 4);
    line20 = _mm_loadu_ps(line2);
    line21 = _mm_loadu_ps(line2 + 4);
    max12 = _mm_max_ps(_mm_max_ps(line10, line20), _mm_max_ps(line11, line21));
    _mm_storeu_ps(out_ptr, max12);
    out_ptr += 4;
    line1 += 8;
    line2 += 8;

    //  const float* line0 = input + c * in_hw;
    //     for (int i = 0; i < mid_h; i++)
    //     {
    //         // left
    //         float max0 = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line2[0], line2[1]));
    //         *out_ptr = arm64_max(arm64_max(line0[0], line0[1]), max0);
    //         out_ptr++;

    //         // mid
    //         for (int j = 0; j < mid_w; j++)
    //         {
    //             float max0 = arm64_max(arm64_max(line0[0], line0[1]), line0[2]);
    //             float max1 = arm64_max(arm64_max(line1[0], line1[1]), line1[2]);
    //             float max2 = arm64_max(arm64_max(line2[0], line2[1]), line2[2]);
    //             *out_ptr = arm64_max(arm64_max(max0, max1), max2);
    //             out_ptr++;
    //             line0 += 1;
    //             line1 += 1;
    //             line2 += 1;
    //         }
    //         max0 = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line2[0], line2[1]));
    //         *out_ptr = arm64_max(arm64_max(line0[0], line0[1]), max0);
    //         out_ptr++;
    //         line0 += 2;
    //         line1 += 2;
    //         line2 += 2;
    //     }

    const float* line0 = input;
    __m128 max0;
    __m128 line00;
    __m128 line01;
    __m128 line02;
    for (int i = 0; i < mid_h; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        max1 = _mm_max_ps(line10, line11);
        max2 = _mm_max_ps(line20, line21);
        max0 = _mm_max_ps(line00, line01);
        max = _mm_max_ps(_mm_max_ps(max0, max1), max2);
        _mm_storeu_ps(out_ptr, max);
        out_ptr += 4;

        for (int j = 0; j < mid_w; j++)
        {
            /* code */
            // for (int j = 0; j < mid_w; j++)
            // {
            //     float max0 = arm64_max(arm64_max(line0[0], line0[1]), line0[2]);
            //     float max1 = arm64_max(arm64_max(line1[0], line1[1]), line1[2]);
            //     float max2 = arm64_max(arm64_max(line2[0], line2[1]), line2[2]);
            //     *out_ptr = arm64_max(arm64_max(max0, max1), max2);
            //     out_ptr++;
            //     line0 += 1;
            //     line1 += 1;
            //     line2 += 1;
            // }
            // max0 = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line2[0], line2[1]));
            // *out_ptr = arm64_max(arm64_max(line0[0], line0[1]), max0);
            // out_ptr++;
            // line0 += 2;
            // line1 += 2;
            // line2 += 2;

            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            line22 = _mm_loadu_ps(line2 + 8);
            max0 = _mm_max_ps(_mm_max_ps(line00, line01), line02);
            max1 = _mm_max_ps(_mm_max_ps(line10, line11), line12);
            max2 = _mm_max_ps(_mm_max_ps(line20, line21), line22);
            _mm_storeu_ps(out_ptr, _mm_max_ps(_mm_max_ps(max0, max1), max2));

            out_ptr += 4;
            line0 += 4;
            line1 += 4;
            line2 += 4;
        }

        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        max0 = _mm_max_ps(line00, line01);
        max1 = _mm_max_ps(line10, line11);
        max2 = _mm_max_ps(line20, line21);
        _mm_storeu_ps(out_ptr, _mm_max_ps(_mm_max_ps(max0, max1), max2));
        out_ptr += 4;
        line0 += 8;
        line1 += 8;
        line2 += 8;
    }

    // *out_ptr = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line0[0], line0[1]));
    //     out_ptr++;

    //     for (int j = 0; j < mid_w; j++)
    //     {
    //         float max0 = arm64_max(arm64_max(line0[0], line0[1]), line0[2]);
    //         float max1 = arm64_max(arm64_max(line1[0], line1[1]), line1[2]);

    //         *out_ptr = arm64_max(max0, max1);
    //         out_ptr++;
    //         line0 += 1;
    //         line1 += 1;
    //     }

    //     *out_ptr = arm64_max(arm64_max(line1[0], line1[1]), arm64_max(line0[0], line0[1]));

    line00 = _mm_loadu_ps(line0);
    line01 = _mm_loadu_ps(line0 + 4);
    line10 = _mm_loadu_ps(line1);
    line11 = _mm_loadu_ps(line1 + 4);
    max0 = _mm_max_ps(line00, line01);
    max1 = _mm_max_ps(line10, line11);
    _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));
    out_ptr += 4;
    for (int i = 0; i < mid_w; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        line02 = _mm_loadu_ps(line0 + 8);
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line12 = _mm_loadu_ps(line1 + 8);
        max0 = _mm_max_ps(line00, line01);
        max1 = _mm_max_ps(line10, line11);
        max2 = _mm_max_ps(line02, line12);
        _mm_storeu_ps(out_ptr, _mm_max_ps(_mm_max_ps(max0, max1), max2));
        out_ptr += 4;
        line0 += 4;
        line1 += 4;
    }
    line00 = _mm_loadu_ps(line0);
    line01 = _mm_loadu_ps(line0 + 4);
    line10 = _mm_loadu_ps(line1);
    line11 = _mm_loadu_ps(line1 + 4);
    max0 = _mm_max_ps(line00, line01);
    max1 = _mm_max_ps(line10, line11);
    _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));
}

static void max_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    const float* line0 = input;
    const float* line1 = input + inw * 4;
    const float* line2 = input + inw * 8;
    float* out_ptr = output;

    __m128 line00;
    __m128 line01;
    __m128 line02;
    __m128 line10;
    __m128 line11;
    __m128 line12;
    __m128 line20;
    __m128 line21;
    __m128 line22;

    __m128 max0;
    __m128 max1;
    __m128 max2;
    __m128 max;

    int remain_w = inw - 2 * outw;

    for (int i = 0; i < outh; i++)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            line22 = _mm_loadu_ps(line2 + 8);
            max0 = _mm_max_ps(_mm_max_ps(line00, line01), line02);
            max1 = _mm_max_ps(_mm_max_ps(line10, line11), line12);
            max2 = _mm_max_ps(_mm_max_ps(line20, line21), line22);

            _mm_storeu_ps(out_ptr, _mm_max_ps(_mm_max_ps(max0, max1), max2));

            line0 += 8;
            line1 += 8;
            line2 += 8;
            out_ptr += 4;
        }

        if (pad_w1 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            max0 = _mm_max_ps(line00, line01);
            max1 = _mm_max_ps(line10, line11);
            max2 = _mm_max_ps(line20, line21);
            _mm_storeu_ps(out_ptr, _mm_max_ps(_mm_max_ps(max0, max1), max2));
            out_ptr += 4;
        }

        line0 += (remain_w + inw) * 4;
        line1 += (remain_w + inw) * 4;
        line2 += (remain_w + inw) * 4;
    }

    if (pad_h1 == 1)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            max0 = _mm_max_ps(_mm_max_ps(line00, line01), line02);
            max1 = _mm_max_ps(_mm_max_ps(line10, line11), line12);

            _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));

            line0 += 8;
            line1 += 8;
            line2 += 8;
            out_ptr += 4;
        }

        if (pad_w1 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            max0 = _mm_max_ps(line00, line01);
            max1 = _mm_max_ps(line10, line11);
            _mm_storeu_ps(out_ptr, _mm_max_ps(max0, max1));
        }
    }
}

static void avg_3x3s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int loopw = (inw - 2) >> 1;
    int looph = outh - 1;

    if (is_caffe == 1 || inw % 2 == 1)
    {
        outw--;
    }
    if (is_caffe == 1 || inh % 2 == 1)
        outh--;

    int remain_w = inw - loopw * 2 + 1;

    if (is_caffe == 1)
    {
        remain_w = 1;
    }

    __m128 scalar_011 = _mm_set1_ps(0.11111111f);
    __m128 scalar_016 = _mm_set1_ps(0.16666667f);
    __m128 scalar_033 = _mm_set1_ps(0.3333333f);
    __m128 scalar_025 = _mm_set1_ps(0.25f);

    const float* line1 = input;
    const float* line2 = input + inw * 4;
    float* out_ptr = output;
    __m128 line10 = _mm_loadu_ps(line1);
    __m128 line11 = _mm_loadu_ps(line1 + 4);
    __m128 line20 = _mm_loadu_ps(line2);
    __m128 line21 = _mm_loadu_ps(line2 + 4);
    __m128 sum1 = _mm_add_ps(line10, line11);
    __m128 sum2 = _mm_add_ps(line20, line21);
    __m128 sum = _mm_add_ps(sum1, sum2);
    if (is_caffe == 0)
    {
        sum = _mm_mul_ps(sum, scalar_025);
    }
    else
    {
        sum = _mm_mul_ps(sum, scalar_011);
    }
    _mm_storeu_ps(out_ptr, sum);
    line1 += 4;
    line2 += 4;
    out_ptr += 4;

    __m128 line12;
    __m128 line22;
    for (int j = 0; j < loopw; j++)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line12 = _mm_loadu_ps(line1 + 8);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        line22 = _mm_loadu_ps(line2 + 8);

        sum1 = _mm_add_ps(line10, _mm_add_ps(line11, line12));
        sum2 = _mm_add_ps(line20, _mm_add_ps(line21, line22));
        sum = _mm_add_ps(sum1, sum2);

        if (is_caffe == 0)
        {
            sum = _mm_mul_ps(sum, scalar_016);
        }
        else
        {
            sum = _mm_mul_ps(sum, scalar_011);
        }

        _mm_storeu_ps(out_ptr, sum);
        line1 += 8;
        line2 += 8;
        out_ptr += 4;
    }

    if (inw % 2 == 1)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        sum1 = _mm_add_ps(line10, line11);
        sum2 = _mm_add_ps(line20, line21);
        sum = _mm_add_ps(sum1, sum2);
        if (is_caffe == 0)
        {
            sum = _mm_mul_ps(sum, scalar_025);
        }
        else
        {
            sum = _mm_mul_ps(sum, scalar_011);
        }
        _mm_storeu_ps(out_ptr, sum);
        out_ptr += 4;
    }
    else if (inw % 2 == 0 && is_caffe == 1)
    {
        // line10 = _mm_loadu_ps(line1);
        // line20 = _mm_loadu_ps(line2);
        // sum = _mm_add_ps(line10, line20);
        // sum = _mm_mul_ps(sum, scalar_016);
        // _mm_storeu_ps(out_ptr, sum);
        // out_ptr += 4;
    }
    line1 += remain_w * 4;
    line2 += remain_w * 4;

    const float* line0 = line1;
    line1 = line2;
    line2 = line1 + inw * 4;
    __m128 line00;
    __m128 line01;
    __m128 line02;
    __m128 sum0;
    for (int i = 0; i < looph; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        sum0 = _mm_add_ps(line00, line01);
        sum1 = _mm_add_ps(line10, line11);
        sum2 = _mm_add_ps(line20, line21);
        sum = _mm_add_ps(_mm_add_ps(sum0, sum1), sum2);
        if (is_caffe == 0)
        {
            sum = _mm_mul_ps(sum, scalar_016);
        }
        else
        {
            sum = _mm_mul_ps(sum, scalar_011);
        }
        _mm_storeu_ps(out_ptr, sum);
        line0 += 4;
        line1 += 4;
        line2 += 4;
        out_ptr += 4;

        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            line22 = _mm_loadu_ps(line2 + 8);
            sum0 = _mm_add_ps(line00, _mm_add_ps(line01, line02));
            sum1 = _mm_add_ps(line10, _mm_add_ps(line11, line12));
            sum2 = _mm_add_ps(line20, _mm_add_ps(line21, line22));
            sum = _mm_add_ps(sum0, _mm_add_ps(sum1, sum2));
            sum = _mm_mul_ps(sum, scalar_011);
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
            line0 += 8;
            line1 += 8;
            line2 += 8;
        }

        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            sum1 = _mm_add_ps(line10, line11);
            sum2 = _mm_add_ps(line20, line21);
            sum0 = _mm_add_ps(line00, line01);
            sum = _mm_add_ps(sum0, _mm_add_ps(sum1, sum2));
            if (is_caffe == 0)
            {
                sum = _mm_mul_ps(sum, scalar_016);
            }
            else
            {
                sum = _mm_mul_ps(sum, scalar_011);
            }
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            // line00 = _mm_loadu_ps(line0);
            // line10 = _mm_loadu_ps(line1);
            // line20 = _mm_loadu_ps(line2);
            // sum = _mm_add_ps(line00, _mm_add_ps(line10, line20));
            // sum = _mm_mul_ps(sum, scalar_016);
            // _mm_storeu_ps(out_ptr, sum);
            // out_ptr += 4;
        }

        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
        line2 += (inw + remain_w) * 4;
    }

    if (inh % 2 == 1)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        sum1 = _mm_add_ps(line10, line11);
        sum0 = _mm_add_ps(line00, line01);
        sum = _mm_add_ps(sum0, sum1);
        if (is_caffe == 0)
        {
            sum = _mm_mul_ps(sum, scalar_025);
        }
        else
        {
            sum = _mm_mul_ps(sum, scalar_011);
        }
        _mm_storeu_ps(out_ptr, sum);
        out_ptr += 4;
        line0 += 4;
        line1 += 4;
        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            sum0 = _mm_add_ps(line00, _mm_add_ps(line01, line02));
            sum1 = _mm_add_ps(line10, _mm_add_ps(line11, line12));
            sum = _mm_add_ps(sum0, sum1);
            if (is_caffe == 0)
            {
                sum = _mm_mul_ps(sum, scalar_016);
            }
            else
            {
                sum = _mm_mul_ps(sum, scalar_011);
            }
            _mm_storeu_ps(out_ptr, sum);
            line0 += 8;
            line1 += 8;
            out_ptr += 4;
        }

        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            sum0 = _mm_add_ps(line00, line01);
            sum1 = _mm_add_ps(line10, line11);
            sum = _mm_add_ps(sum0, sum1);
            if (is_caffe == 0)
            {
                sum = _mm_mul_ps(sum, scalar_025);
            }
            else
            {
                sum = _mm_mul_ps(sum, scalar_011);
            }
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            // line00 = _mm_loadu_ps(line0);
            // line10 = _mm_loadu_ps(line1);
            // sum = _mm_add_ps(line00, line10);
            // sum = _mm_mul_ps(sum, scalar_016);
            // _mm_storeu_ps(out_ptr, sum);
            // out_ptr += 4;
        }
    }
    else if (inh % 2 == 0 && is_caffe == 1)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        sum = _mm_add_ps(line00, line01);
        sum = _mm_mul_ps(sum, scalar_016);
        _mm_storeu_ps(out_ptr, sum);
        line0 += 4;
        out_ptr += 4;
        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            sum = _mm_add_ps(line00, _mm_add_ps(line01, line02));
            sum = _mm_mul_ps(sum, scalar_016);
            _mm_storeu_ps(out_ptr, sum);
            line0 += 8;
            out_ptr += 4;
        }
        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            sum = _mm_add_ps(line00, line01);
            sum = _mm_mul_ps(sum, scalar_016);
            _mm_storeu_ps(out_ptr, sum);
        }
        else if (inw % 2 == 0)
        {
            // sum = _mm_loadu_ps(line0);
            // sum = _mm_mul_ps(sum, scalar_025);
            // _mm_storeu_ps(out_ptr, sum);
        }
    }
}

static void max_3x3s2_p1(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                         int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (is_caffe == 1 || inw % 2 == 1)
    {
        outw--;
    }
    if (is_caffe == 1 || inh % 2 == 1)
        outh--;

    int loopw = outw - 1;
    int looph = outh - 1;
    int remain_w = inw - outw * 2 + 1;

    const float* line1 = input;
    const float* line2 = input + inw * 4;
    float* out_ptr = output;
    __m128 line10 = _mm_loadu_ps(line1);
    __m128 line11 = _mm_loadu_ps(line1 + 4);
    __m128 line20 = _mm_loadu_ps(line2);
    __m128 line21 = _mm_loadu_ps(line2 + 4);
    __m128 max1 = _mm_max_ps(line10, line11);
    __m128 max2 = _mm_max_ps(line20, line21);
    __m128 max = _mm_max_ps(max1, max2);
    _mm_storeu_ps(out_ptr, max);
    line1 += 4;
    line2 += 4;
    out_ptr += 4;

    __m128 line12;
    __m128 line22;
    for (int j = 0; j < loopw; j++)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line12 = _mm_loadu_ps(line1 + 8);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        line22 = _mm_loadu_ps(line2 + 8);

        max1 = _mm_max_ps(line10, _mm_max_ps(line11, line12));
        max2 = _mm_max_ps(line20, _mm_max_ps(line21, line22));
        max = _mm_max_ps(max1, max2);
        _mm_storeu_ps(out_ptr, max);

        line1 += 8;
        line2 += 8;
        out_ptr += 4;
    }

    if (inw % 2 == 1)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        max1 = _mm_max_ps(line10, line11);
        max2 = _mm_max_ps(line20, line21);
        max = _mm_max_ps(max1, max2);
        _mm_storeu_ps(out_ptr, max);
        out_ptr += 4;
    }
    else if (inw % 2 == 0 && is_caffe == 1)
    {
        line10 = _mm_loadu_ps(line1);
        line20 = _mm_loadu_ps(line2);
        _mm_storeu_ps(out_ptr, _mm_max_ps(line10, line20));
        out_ptr += 4;
    }
    line1 += remain_w * 4;
    line2 += remain_w * 4;

    const float* line0 = line1;
    line1 = line2;
    line2 = line1 + inw * 4;
    __m128 line00;
    __m128 line01;
    __m128 line02;
    __m128 max0;
    for (int i = 0; i < looph; i++)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line20 = _mm_loadu_ps(line2);
        line21 = _mm_loadu_ps(line2 + 4);
        max0 = _mm_max_ps(line00, line01);
        max1 = _mm_max_ps(line10, line11);
        max2 = _mm_max_ps(line20, line21);
        max = _mm_max_ps(_mm_max_ps(max0, max1), max2);
        _mm_storeu_ps(out_ptr, max);
        line0 += 4;
        line1 += 4;
        line2 += 4;
        out_ptr += 4;

        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            line22 = _mm_loadu_ps(line2 + 8);
            max0 = _mm_max_ps(line00, _mm_max_ps(line01, line02));
            max1 = _mm_max_ps(line10, _mm_max_ps(line11, line12));
            max2 = _mm_max_ps(line20, _mm_max_ps(line21, line22));
            max = _mm_max_ps(max0, _mm_max_ps(max1, max2));
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
            line0 += 8;
            line1 += 8;
            line2 += 8;
        }

        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            max1 = _mm_max_ps(line10, line11);
            max2 = _mm_max_ps(line20, line21);
            max0 = _mm_max_ps(line00, line01);
            max = _mm_max_ps(max0, _mm_max_ps(max1, max2));
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line10 = _mm_loadu_ps(line1);
            line20 = _mm_loadu_ps(line2);
            max = _mm_max_ps(line00, _mm_max_ps(line10, line20));
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
        }

        line0 += (inw + remain_w) * 4;
        line1 += (inw + remain_w) * 4;
        line2 += (inw + remain_w) * 4;
    }

    if (inh % 2 == 1)
    {
        line10 = _mm_loadu_ps(line1);
        line11 = _mm_loadu_ps(line1 + 4);
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        max1 = _mm_max_ps(line10, line11);
        max0 = _mm_max_ps(line00, line01);
        max = _mm_max_ps(max0, max1);
        _mm_storeu_ps(out_ptr, max);
        out_ptr += 4;
        line0 += 4;
        line1 += 4;
        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            max0 = _mm_max_ps(line00, _mm_max_ps(line01, line02));
            max1 = _mm_max_ps(line10, _mm_max_ps(line11, line12));
            max = _mm_max_ps(max0, max1);
            _mm_storeu_ps(out_ptr, max);
            line0 += 8;
            line1 += 8;
            out_ptr += 4;
        }

        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            max0 = _mm_max_ps(line00, line01);
            max1 = _mm_max_ps(line10, line11);
            max = _mm_max_ps(max0, max1);
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
        }
        else if (inw % 2 == 0 && is_caffe == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line10 = _mm_loadu_ps(line1);
            max = _mm_max_ps(line00, line10);
            _mm_storeu_ps(out_ptr, max);
            out_ptr += 4;
        }
    }
    else if (inh % 2 == 0 && is_caffe == 1)
    {
        line00 = _mm_loadu_ps(line0);
        line01 = _mm_loadu_ps(line0 + 4);
        max = _mm_max_ps(line00, line01);
        _mm_storeu_ps(out_ptr, max);
        line0 += 4;
        out_ptr += 4;
        for (int j = 0; j < loopw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            max = _mm_max_ps(line00, _mm_max_ps(line01, line02));
            _mm_storeu_ps(out_ptr, max);
            line0 += 8;
            out_ptr += 4;
        }
        if (inw % 2 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            max = _mm_max_ps(line00, line01);
            _mm_storeu_ps(out_ptr, max);
        }
        else if (inw % 2 == 0)
        {
            max = _mm_loadu_ps(line0);
            _mm_storeu_ps(out_ptr, max);
        }
    }
}

static void avg_3x3s2(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                      int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    if (pad_w1 > 0)
    {
        outw--;
    }
    if (pad_h1 > 0)
    {
        outh--;
    }
    const float* line0 = input;
    const float* line1 = input + inw * 4;
    const float* line2 = input + inw * 8;
    float* out_ptr = output;

    __m128 scalar_011 = _mm_set1_ps(0.11111111f);
    __m128 scalar_016 = _mm_set1_ps(0.16666667f);
    __m128 scalar_025 = _mm_set1_ps(0.25f);
    __m128 scalar_05 = _mm_set1_ps(0.5f);
    __m128 scalar_033 = _mm_set1_ps(0.33333333f);

    __m128 line00;
    __m128 line01;
    __m128 line02;
    __m128 line10;
    __m128 line11;
    __m128 line12;
    __m128 line20;
    __m128 line21;
    __m128 line22;

    __m128 sum0;
    __m128 sum1;
    __m128 sum2;
    __m128 sum;

    int remain_w = inw - 2 * outw;

    for (int i = 0; i < outh; i++)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            line22 = _mm_loadu_ps(line2 + 8);
            sum0 = _mm_add_ps(_mm_add_ps(line00, line01), line02);
            sum1 = _mm_add_ps(_mm_add_ps(line10, line11), line12);
            sum2 = _mm_add_ps(_mm_add_ps(line20, line21), line22);
            sum = _mm_add_ps(_mm_add_ps(sum0, sum1), sum2);
            sum = _mm_mul_ps(sum, scalar_011);
            _mm_storeu_ps(out_ptr, sum);

            line0 += 8;
            line1 += 8;
            line2 += 8;
            out_ptr += 4;
        }

        if (pad_w1 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line20 = _mm_loadu_ps(line2);
            line21 = _mm_loadu_ps(line2 + 4);
            sum0 = _mm_add_ps(line00, line01);
            sum1 = _mm_add_ps(line10, line11);
            sum2 = _mm_add_ps(line20, line21);
            sum = _mm_add_ps(_mm_add_ps(sum0, sum1), sum2);
            sum = _mm_mul_ps(sum, scalar_016);
            _mm_storeu_ps(out_ptr, sum);
            out_ptr += 4;
        }

        line0 += (remain_w + inw) * 4;
        line1 += (remain_w + inw) * 4;
        line2 += (remain_w + inw) * 4;
    }

    if (pad_h1 == 1)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            line12 = _mm_loadu_ps(line1 + 8);
            sum0 = _mm_add_ps(_mm_add_ps(line00, line01), line02);
            sum1 = _mm_add_ps(_mm_add_ps(line10, line11), line12);
            sum = _mm_add_ps(sum0, sum1);
            sum = _mm_mul_ps(sum, scalar_016);
            _mm_storeu_ps(out_ptr, sum);

            line0 += 8;
            line1 += 8;
            out_ptr += 4;
        }

        if (pad_w1 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line10 = _mm_loadu_ps(line1);
            line11 = _mm_loadu_ps(line1 + 4);
            sum0 = _mm_add_ps(line00, line01);
            sum1 = _mm_add_ps(line10, line11);
            sum = _mm_add_ps(sum0, sum1);
            sum = _mm_mul_ps(sum, scalar_025);
            _mm_storeu_ps(out_ptr, sum);
        }
        else if (pad_w1 == 2)
        {
            line00 = _mm_loadu_ps(line0);
            line10 = _mm_loadu_ps(line1);
            sum = _mm_add_ps(line00, line10);
            sum = _mm_mul_ps(sum, scalar_05);
            _mm_storeu_ps(out_ptr, sum);
        }
    }
    else if (pad_h1 == 2)
    {
        for (int j = 0; j < outw; j++)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            line02 = _mm_loadu_ps(line0 + 8);
            sum0 = _mm_add_ps(_mm_add_ps(line00, line01), line02);
            sum0 = _mm_mul_ps(sum0, scalar_033);
            _mm_storeu_ps(out_ptr, sum0);
            line0 += 8;
            out_ptr += 4;
        }

        if (pad_w1 == 1)
        {
            line00 = _mm_loadu_ps(line0);
            line01 = _mm_loadu_ps(line0 + 4);
            sum0 = _mm_add_ps(line00, line01);
            sum0 = _mm_mul_ps(sum0, scalar_05);
            _mm_storeu_ps(out_ptr, sum0);
        }
        else if (pad_w1 == 2)
        {
            line00 = _mm_loadu_ps(line0);
            _mm_storeu_ps(out_ptr, line00);
        }
    }
}

static void avg_global(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                       int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        float sum = 0.f;
        for (int j = 0; j < block; j++)
        {
            __m128 p00 = _mm_loadu_ps(line0);
            __m128 p01 = _mm_loadu_ps(line0 + 4);
            p00 = _mm_add_ps(p00, p01);


#ifdef _WIN32
            sum += (p00.m128_f32[0] + p00.m128_f32[1] + p00.m128_f32[2] + p00.m128_f32[3]);
#else
            sum += (p00[0] + p00[1] + p00[2] + p00[3]);
#endif
            line0 += 8;
        }
        for (int j = tail; j < in_hw; j++)
        {
            sum += line0[0];
            line0++;
        }
        *out_ptr = sum / in_hw;
    }
}
static void max_global(const float* input, float* output, int inc, int inh, int inw, int outh, int outw, int k_h,
                       int k_w, int s_h, int s_w, int pad_h0, int pad_w0, int pad_h1, int pad_w1, int is_caffe)
{
    int in_hw = inw * inh;
    int block = in_hw >> 3;
    int tail = in_hw & ~7;

    for (int c = 0; c < inc; c++)
    {
        const float* line0 = input + c * in_hw;
        float* out_ptr = output + c;
        __m128 p00 = _mm_loadu_ps(line0);
        __m128 res = p00;
        for (int j = 0; j < block; j++)
        {
            __m128 p00 = _mm_loadu_ps(line0);
            __m128 p01 = _mm_loadu_ps(line0 + 4);
            __m128 max0 = _mm_max_ps(p00, p01);
            res = _mm_max_ps(res, max0);
            line0 += 8;
        }
#ifdef _WIN32
        float max_ = max(max(res.m128_f32[0], res.m128_f32[1]), max(res.m128_f32[2], res.m128_f32[3]));
#else
        float max_ = max(max(res[0], res[1]), max(res[2], res[3]));
#endif
        for (int j = tail; j < in_hw; j++)
        {
            max_ = max(max_, line0[0]);
            line0++;
        }
        *out_ptr = max_;
    }
}

int pooling_kernel_perf_prerun(struct ir_tensor* input, struct ir_tensor* out, struct pool_param* param)
{
    int pool_size = POOL_GENERIC;

    /* global pooling */
    if (param->global)
    {
        if (param->pool_method == POOL_AVG)
            param->funct = ( pooling_kernel_t )avg_global;
        else if (param->pool_method == POOL_MAX)
            param->funct = ( pooling_kernel_t )max_global;

        assert(param->funct != NULL);
        return 0;
    }

    /* general pooling */
    if (param->stride_h == 2 && param->stride_w == 2)
    {
        if (param->kernel_h == 2 && param->kernel_w == 2)
            pool_size = POOL_K2S2;
        else if (param->kernel_h == 3 && param->kernel_w == 3)
            pool_size = POOL_K3S2;
    }
    else if (param->stride_h == 1 && param->stride_w == 1)
    {
        if (param->kernel_h == 3 && param->kernel_w == 3)
            pool_size = POOL_K3S1;
    }

    int pool_method;    // 0:max    1:avg
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h0;
    int pad_h1;
    int pad_w0;
    int pad_w1;
    int global;    // 0:general    1:global
    int caffe_flavor;

    /* general max pooling, k2s2, k2k2p1, k3s1p1, k3s2, k3s2p1 */
    if (param->pool_method == POOL_MAX)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (param->pad_h0 == 0)
            {
                if (pool_size == POOL_K2S2)
                {
                    param->funct = ( pooling_kernel_t )max_2x2s2;
                }
                else if (pool_size == POOL_K3S2)
                {
                    param->funct = ( pooling_kernel_t )max_3x3s2;
                }
            }
            else if (param->pad_h0 == 1)
            {
                if (pool_size == POOL_K2S2)
                {
                    param->funct = ( pooling_kernel_t )max_2x2s2_p1;
                }
                else if (pool_size == POOL_K3S2)
                {
                    param->funct = ( pooling_kernel_t )max_3x3s2_p1;
                }
                else if (pool_size == POOL_K3S1)
                {
                    param->funct = ( pooling_kernel_t )max_3x3s1_p1;
                }
            }
        }

        if (param->funct != NULL)
            return 0;
        else
        {
            fprintf(stderr, "perf general max pooling func not be find\n");
            return -1;
        }
    }

    /* general avg pooling, k2s2, k2s2p1, k3s2, k3s2p1 */
    if (param->pool_method == POOL_AVG)
    {
        if ((param->pad_h0 == param->pad_w0) && (param->pad_h1 == param->pad_w1))
        {
            if (param->pad_h0 == 0 && param->pad_h1 == 0)
            {
                if (pool_size == POOL_K2S2)
                {
                    param->funct = ( pooling_kernel_t )avg_2x2s2;
                }
                else if (pool_size == POOL_K3S2)
                {
                    param->funct = ( pooling_kernel_t )avg_3x3s2;
                }
            }
            else if (param->pad_h0 == 1 && param->pad_h1 == 1)
            {
                if (pool_size == POOL_K2S2)
                {
                    param->funct = ( pooling_kernel_t )avg_2x2s2_p1;
                }
                else if (pool_size == POOL_K3S2)
                {
                    param->funct = ( pooling_kernel_t )avg_3x3s2_p1;
                }
            }
        }

        if (param->funct != NULL)
            return 0;
        else
        {
            fprintf(stderr, "perf general avg pooling func not be find\n");
            return -1;
        }
    }

    fprintf(stderr, "perf pooling func not be find\n");
    return -1;
}

#define PACK4 4

static void pack4(float* input, float* input_buffer, int in_h, int in_w)
{
    for (size_t i = 0; i < in_h; i++)
    {
        for (int j = 0; j < in_w; j++)
        {
            for (int c = 0; c < PACK4; c++)
            {
                input_buffer[i * in_w * PACK4 + j * PACK4 + c] = input[(unsigned long)c * in_w * in_h + i * in_w + j];
            }
        }
    }
}

static void unpack4(float* output_buffer, float* output, int out_h, int out_w)
{
    for (size_t i = 0; i < PACK4; i++)
    {
        for (size_t j = 0; j < out_h; j++)
        {
            for (size_t k = 0; k < out_w; k++)
            {
                output[i * out_h * out_w + j * out_w + k] = output_buffer[j * out_w * PACK4 + k * PACK4 + i];
            }
        }
    }
}

int pooling_kernel_perf_run(struct ir_tensor* input, struct ir_tensor* output, struct pool_param* param, int num_thread)
{
    // fprintf(stderr, "perf pooling_kernel_run\n");
    int is_caffe = param->caffe_flavor;
    pooling_kernel_t kernel = (pooling_kernel_t)(param->funct);

    int batch = input->dims[0];
    int c = input->dims[1];
    int in_h = input->dims[2];
    int in_w = input->dims[3];

    int out_h = output->dims[2];
    int out_w = output->dims[3];

    int img_size = c * in_h * in_w;
    int feature_size = c * out_h * out_w;

    if (param->global)
    {
        for (int n = 0; n < batch; n++)
        {
            float* input_frame = ( float* )input->data + n * img_size;
            float* output_frame = ( float* )output->data + n * feature_size;
#pragma omp parallel for num_threads(num_thread)
            for (int ch = 0; ch < c; ch++)
            {
                float* cur_input = input_frame + ch * in_h * in_w;
                float* cur_output = output_frame + ch * out_h * out_w;
                kernel(cur_input, cur_output, 1, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
                       param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->pad_h1, param->pad_w1,
                       is_caffe);
            }
        }
    }
    else
    {
        int packc4 = c >> 2;
        float* input_buffer  = ( float* )calloc(sizeof(float), PACK4 * (size_t)in_h * in_w);
        float* output_buffer = ( float* )calloc(sizeof(float), PACK4 * (size_t)out_h * out_w);
        for (int n = 0; n < batch; n++)
        {
            for (int pck = 0; pck < packc4; pck++)
            {
                float* input_cur = ( float* )input->data + n * img_size + pck * PACK4 * in_h * in_w;
                float* output_cur = ( float* )output->data + n * feature_size + pck * PACK4 * out_h * out_w;
                pack4(input_cur, input_buffer, in_h, in_w);
                kernel(input_buffer, output_buffer, c, in_h, in_w, out_h, out_w, param->kernel_h, param->kernel_w,
                       param->stride_h, param->stride_w, param->pad_h0, param->pad_w0, param->pad_h1, param->pad_w1,
                       is_caffe);
                unpack4(output_buffer, output_cur, out_h, out_w);
            }
        }

        free(input_buffer);
        free(output_buffer);
    }

    return 0;
}