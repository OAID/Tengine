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
 */

#include <stdint.h>
#include <math.h>
#include <string.h>
#include "tengine_operations.h"
#include "stb_image.h"
#include "stb_image_write.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))

int check_file_exist(const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
    {
        fprintf(stderr, "Input file not existed: %s\n", file_name);
        return 0;
    }
    fclose(fp);
    return 1;
}

image load_image_stb(const char* filename, int channels)
{
    int w, h, c;
    unsigned char* data = stbi_load(filename, &w, &h, &c, channels);

    if (!data)
    {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels)
        c = channels;
    int src_c = c;
    if (c > 3)
        c = 3;
    image im = make_image(w, h, c);
    for (int k = 0; k < c; ++k)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                int dst_index = i + w * j + w * h * k;
                int src_index = k + src_c * i + src_c * w * j;
                im.data[dst_index] = (float)data[src_index];
            }
        }
    }

    free(data);
    return im;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = (float*)calloc((size_t)h * w * c, sizeof(float));
    return out;
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image imread2caffe(image resImg, int img_w, int img_h, float* means, float* scale)
{
    for (int c = 0; c < resImg.c; c++)
    {
        for (int i = 0; i < resImg.h; i++)
        {
            for (int j = 0; j < resImg.w; j++)
            {
                int index = c * resImg.h * resImg.w + i * resImg.w + j;
                resImg.data[index] = (resImg.data[index] - means[c]) * scale[c];
            }
        }
    }
    return resImg;
}

image imread_process(const char* filename, int img_w, int img_h, float* means, float* scale)
{
    image out = imread(filename);

    int choice = 0;
    if (out.c == 1)
        choice = 0;
    else
        choice = 2;

    switch (choice)
    {
    case 0:
        out = gray2bgr(out);
        break;
    case 1:
        out = rgb2gray(out);
        break;
    case 2:
        out = rgb2bgr_permute(out);
        break;
    default:
        break;
    }

    image resImg = make_image(img_w, img_h, out.c);

    tengine_resize_f32(out.data, resImg.data, img_w, img_h, out.c, out.h, out.w);
    resImg = imread2caffe(resImg, img_w, img_h, means, scale);

    free_image(out);
    return resImg;
}

static double get_pixelData(image m, int x, int y, int c)
{
    if (x < m.w && y < m.h && c < m.c)
    {
        return m.data[c * m.h * m.w + y * m.w + x];
    }
    else
    {
        return 0;
    }
}
// Bilinear Inter
image resize_image(image im, int ow, int oh)
{
    image resized = make_image(ow, oh, im.c);
    memset(resized.data, 0, sizeof(float) * ow * oh * im.c);
    float* tmpData = resized.data;

#ifdef __ARM_NEON
#ifdef __aarch64__
    int c = im.c;
    int h = im.h;
    int w = im.w;
    float shift = 0.f;
    float _scale_x = (float)((w - shift) / (ow - shift));
    float _scale_y = (float)((h - shift) / (oh - shift));
    float32x4_t scale_x = vdupq_n_f32(_scale_x);
    float offset = 0.5;
    int in_hw = h * w;
    const float32x4_t offset_1 = vdupq_n_f32(1.f);
    const float32x4_t offset_n1 = vdupq_n_f32(0.f);
    const float32x4_t offset_half = vdupq_n_f32(offset);
    const float32x4_t maxX = vdupq_n_f32(0);
    const float32x4_t minX = vdupq_n_f32(w - 1);
    int32x4_t w_0 = vdupq_n_s32(w);
    int32x4_t in_hw_0 = vdupq_n_s32(in_hw);

    for (int k = 0; k < c; k++)
    {
        int32x4_t k_0 = vdupq_n_s32(k);
        for (int j = 0; j < oh; j++)
        {
            float fy = (j + offset) * _scale_y - offset;
            int sy = floor(fy);
            fy -= sy;
            sy = T_MIN(sy, h - 2);
            sy = T_MAX(0, sy);
            float32x4_t _fy = vdupq_n_f32(fy);
            float32x4_t fy_0 = vsubq_f32(offset_1, _fy);
            float _fy_0 = 1 - fy;
            int32x4_t sy_0 = vdupq_n_s32(sy);
            for (int i = 0; i < (ow & -4); i += 4)
            {
                float _i_cnt_f[4] = {i + 0.f, i + 1.f, i + 2.f, i + 3.f};
                float32x4_t i_cnt_f = vld1q_f32(_i_cnt_f);
                float32x4_t fx = vsubq_f32(vmulq_f32(vaddq_f32(i_cnt_f, offset_half), scale_x), offset_half);
                int32x4_t sx_0 = vcvtq_s32_f32(fx);
                float32x4_t sx = vcvtq_f32_s32(sx_0);

                fx = vsubq_f32(fx, sx);
                fx = vmaxq_f32(vsubq_f32(vminq_f32(vaddq_f32(fx, sx), minX), sx), maxX);

                fx = vmaxq_f32(fx, maxX);
                sx = vmaxq_f32(sx, offset_n1);
                sx = vminq_f32(sx, minX);

                float32x4_t fx_0 = vsubq_f32(offset_1, fx);

                const int32x4_t in_idx = vaddq_s32(vaddq_s32(vmulq_s32(sy_0, w_0), vcvtq_s32_f32(sx)), vmulq_s32(in_hw_0, k_0));
                int32x4_t in_index0 = in_idx;
                int32x4_t in_index2 = vaddq_s32(in_idx, vcvtq_s32_f32(offset_1));
                int32x4_t in_index1 = vaddq_s32(in_idx, w_0);
                int32x4_t in_index3 = vaddq_s32(vaddq_s32(in_idx, vcvtq_s32_f32(offset_1)), w_0);

                float32x4_t inTemp0 = vdupq_n_f32(0);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 0), inTemp0, 0);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 1), inTemp0, 1);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 2), inTemp0, 2);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 3), inTemp0, 3);

                float32x4_t inTemp1 = vdupq_n_f32(0);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 0), inTemp1, 0);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 1), inTemp1, 1);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 2), inTemp1, 2);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 3), inTemp1, 3);

                float32x4_t inTemp2 = vdupq_n_f32(0);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 0), inTemp2, 0);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 1), inTemp2, 1);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 2), inTemp2, 2);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 3), inTemp2, 3);

                float32x4_t inTemp3 = vdupq_n_f32(0);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 0), inTemp3, 0);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 1), inTemp3, 1);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 2), inTemp3, 2);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 3), inTemp3, 3);

                float32x4_t data0 = vmulq_f32(vmulq_f32(inTemp0, fx_0), fy_0);
                float32x4_t data1 = vmulq_f32(vmulq_f32(inTemp1, fx_0), _fy);
                float32x4_t data2 = vmulq_f32(vmulq_f32(inTemp2, fx), fy_0);
                float32x4_t data3 = vmulq_f32(vmulq_f32(inTemp3, fx), _fy);

                float32x4_t outTemp = vaddq_f32(vaddq_f32(vaddq_f32(data0, data1), data2), data3);
                vst1q_f32(tmpData, outTemp);
                tmpData += 4;
            }
            for (int i = ow & ~3; i < ow; i++)
            {
                float fx = (i + offset) * _scale_x - offset;
                int sx = floor(fx);
                fx -= sx;
                if (sx < 0)
                {
                    sx = 0;
                    fx = 0;
                }
                if (sx >= w - 1)
                {
                    fx = 0;
                    sx = w - 2;
                }
                float fx_0 = 1.f - fx;
                int in_idx = sy * w + sx;
                int in_index = in_idx + k * in_hw;
                float data0 = im.data[in_index] * fx_0 * _fy_0;
                float data1 = im.data[in_index + w] * fx_0 * fy;
                float data2 = im.data[in_index + 1] * fx * _fy_0;
                float data3 = im.data[in_index + w + 1] * fx * fy;

                *tmpData = data0 + data1 + data2 + data3;
                tmpData++;
            }
        }
    }

#else
    int c = im.c;

    int h = im.h;
    int w = im.w;
    float shift = 0.f;
    float _scale_x = (float)((w - shift) / (ow - shift));
    float _scale_y = (float)((h - shift) / (oh - shift));

    float32x4_t scale_x = vdupq_n_f32(_scale_x);
    float offset = 0.5;
    int in_hw = h * w;
    const float32x4_t offset_1 = vdupq_n_f32(1.f);
    const float32x4_t offset_n1 = vdupq_n_f32(0.f);
    const float32x4_t offset_half = vdupq_n_f32(offset);
    const float32x4_t maxX = vdupq_n_f32(0);
    const float32x4_t minX = vdupq_n_f32(w - 1);
    int32x4_t w_0 = vdupq_n_s32(w);
    int32x4_t in_hw_0 = vdupq_n_s32(in_hw);

    for (int k = 0; k < c; k++)
    {
        int32x4_t k_0 = vdupq_n_s32(k);
        for (int j = 0; j < oh; j++)
        {
            float fy = (j + offset) * _scale_y - offset;
            int sy = floor(fy);
            fy -= sy;
            sy = T_MIN(sy, h - 2);
            sy = T_MAX(0, sy);
            float32x4_t _fy = vdupq_n_f32(fy);
            float32x4_t fy_0 = vsubq_f32(offset_1, _fy);
            float _fy_0 = 1.f - fy;
            int32x4_t sy_0 = vdupq_n_s32(sy);

            for (int i = 0; i < (ow & -4); i += 4)
            {
                float _i_cnt_f[4] = {i + 0.f, i + 1.f, i + 2.f, i + 3.f};
                float32x4_t i_cnt_f = vld1q_f32(_i_cnt_f);
                float32x4_t fx = vsubq_f32(vmulq_f32(vaddq_f32(i_cnt_f, offset_half), scale_x), offset_half);
                int32x4_t sx_0 = vcvtq_s32_f32(fx);
                float32x4_t sx = vcvtq_f32_s32(sx_0);
                fx = vsubq_f32(fx, sx);
                fx = vmaxq_f32(vsubq_f32(vminq_f32(vaddq_f32(fx, sx), minX), sx), maxX);

                fx = vmaxq_f32(fx, maxX);
                sx = vmaxq_f32(sx, offset_n1);
                sx = vminq_f32(sx, minX);

                float32x4_t fx_0 = vsubq_f32(offset_1, fx);

                const int32x4_t in_idx = vaddq_s32(vaddq_s32(vmulq_s32(sy_0, w_0), vcvtq_s32_f32(sx)), vmulq_s32(in_hw_0, k_0));

                int32x4_t in_index0 = in_idx;
                int32x4_t in_index2 = vaddq_s32(in_idx, vcvtq_s32_f32(offset_1));
                int32x4_t in_index1 = vaddq_s32(in_idx, w_0);
                int32x4_t in_index3 = vaddq_s32(vaddq_s32(in_idx, vcvtq_s32_f32(offset_1)), w_0);

                float32x4_t inTemp0 = vdupq_n_f32(0);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 0), inTemp0, 0);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 1), inTemp0, 1);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 2), inTemp0, 2);
                inTemp0 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index0, 3), inTemp0, 3);

                float32x4_t inTemp1 = vdupq_n_f32(0);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 0), inTemp1, 0);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 1), inTemp1, 1);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 2), inTemp1, 2);
                inTemp1 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index1, 3), inTemp1, 3);

                float32x4_t inTemp2 = vdupq_n_f32(0);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 0), inTemp2, 0);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 1), inTemp2, 1);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 2), inTemp2, 2);
                inTemp2 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index2, 3), inTemp2, 3);

                float32x4_t inTemp3 = vdupq_n_f32(0);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 0), inTemp3, 0);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 1), inTemp3, 1);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 2), inTemp3, 2);
                inTemp3 = vld1q_lane_f32(im.data + vgetq_lane_s32(in_index3, 3), inTemp3, 3);

                float32x4_t data0 = vmulq_f32(vmulq_f32(inTemp0, fx_0), fy_0);
                float32x4_t data1 = vmulq_f32(vmulq_f32(inTemp1, fx_0), _fy);
                float32x4_t data2 = vmulq_f32(vmulq_f32(inTemp2, fx), fy_0);
                float32x4_t data3 = vmulq_f32(vmulq_f32(inTemp3, fx), _fy);

                float32x4_t outTemp = vaddq_f32(vaddq_f32(vaddq_f32(data0, data1), data2), data3);
                vst1q_f32(tmpData, outTemp);
                tmpData += 4;
            }

            for (int i = (ow & ~3); i < ow; i++)
            {
                float fx = (i + offset) * _scale_x - offset;
                int sx = floor(fx);
                fx -= sx;
                if (sx < 0)
                {
                    sx = 0;
                    fx = 0;
                }
                if (sx >= w - 1)
                {
                    fx = 0;
                    sx = w - 2;
                }
                float fx_0 = 1.f - fx;
                int in_idx = sy * w + sx;
                int in_index = in_idx + k * in_hw;
                float data0 = *(im.data + in_index);
                float data1 = *(im.data + in_index + w);
                float data2 = *(im.data + in_index + 1);
                float data3 = *(im.data + in_index + w + 1);
                *tmpData = data0 * fx_0 * _fy_0 + data1 * fx_0 * fy + data2 * fx * _fy_0 + data3 * fx * fy;

                tmpData++;
            }
        }
    }
#endif

#else
    float scale_x = (float)(im.w) / (ow);
    float scale_y = (float)(im.h) / (oh);
    int w = im.w;
    int h = im.h;
    int in_hw = h * w;
    for (int k = 0; k < im.c; k++)
    {
        for (int j = 0; j < oh; j++)
        {
            float fy = (j + 0.5) * scale_y - 0.5;
            int sy = floor(fy);
            fy -= sy;
            sy = (sy < (h - 2)) ? sy : (h - 2);
            sy = (sy > 0) ? sy : 0;
            float fy_0 = 1.f - fy;

            for (int i = 0; i < ow; i++)
            {
                float fx = (i + 0.5) * scale_x - 0.5;
                int sx = floor(fx);
                fx -= sx;
                if (sx < 0)
                {
                    sx = 0;
                    fx = 0;
                }
                if (sx >= w - 1)
                {
                    fx = 0;
                    sx = w - 2;
                }
                float fx_0 = 1.f - fx;
                int in_idx = sy * w + sx;
                int in_index = in_idx + k * in_hw;
                float data0 = im.data[in_index] * fx_0 * fy_0;
                float data1 = im.data[in_index + w] * fx_0 * fy;
                float data2 = im.data[in_index + 1] * fx * fy_0;
                float data3 = im.data[in_index + w + 1] * fx * fy;

                *tmpData = data0 + data1 + data2 + data3;
                tmpData++;
            }
        }
    }
#endif
    return resized;
}

image copyMaker(image im, int top, int bottom, int left, int right, float value)
{
    int width = im.w + left + right;
    int height = im.h + top + bottom;
    image resImg = make_image(width, height, im.c);
    memset(resImg.data, value, sizeof(float) * width * height * im.c);
    for (int c = 0; c < im.c; c++)
    {
        for (int i = top; i < height - bottom; i++)
        {
            for (int j = left; j < width - right; j++)
            {
                int resIndex = c * height * width + i * width + j;
                int originIndex = c * im.w * im.h + (i - top) * im.w + (j - left);
                resImg.data[resIndex] = im.data[originIndex];
            }
        }
    }

    return resImg;
}

void save_image(image im, const char* name)
{
    char buff[256] = {0};
    unsigned char* data = (unsigned char*)calloc((size_t)im.w * im.h * im.c, sizeof(char));
    int i, k;
    for (k = 0; k < im.c; ++k)
    {
        for (i = 0; i < im.w * im.h; ++i)
        {
            data[i * im.c + k] = (unsigned char)(im.data[i + k * im.w * im.h]);
        }
    }

    int success = 0;
    int f = 0;
    int len = name ? strlen(name) : 0;
    if (3 < len && 0 == strcmp(name + len - 4, ".jpg"))
        f = 1;
    if (3 < len && 0 == strcmp(name + len - 4, ".png"))
        f = 2;
    if (3 < len && 0 == strcmp(name + len - 4, ".tga"))
        f = 3;
    if (3 < len && 0 == strcmp(name + len - 4, ".bmp"))
        f = 4;
    sprintf(buff, "%s", name ? name : "null");

    switch (f)
    {
    case 0:
        strcat(buff, ".jpg");
    case 1:
        success = stbi_write_jpg(buff, im.w, im.h, im.c, data, 80);
        break;
    case 2:
        success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w * im.c);
        break;
    case 3:
        success = stbi_write_tga(buff, im.w, im.h, im.c, data);
        break;
    case 4:
        success = stbi_write_bmp(buff, im.w, im.h, im.c, data);
        break;
    default:
        return;
    }
    free(data);
    if (!success)
        fprintf(stderr, "Failed to write image %s\n", buff);
}

void draw_box(image im, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for (i = 0; i < w; ++i)
    {
        if (x1 < 0)
            x1 = 0;
        if (x1 >= im.w)
            x1 = im.w - 1;
        if (x2 < 0)
            x2 = 0;
        if (x2 >= im.w)
            x2 = im.w - 1;

        if (y1 < 0)
            y1 = 0;
        if (y1 >= im.h)
            y1 = im.h - 1;
        if (y2 < 0)
            y2 = 0;
        if (y2 >= im.h)
            y2 = im.h - 1;

        for (i = x1; i <= x2; ++i)
        {
            im.data[i + y1 * im.w + 0 * im.w * im.h] = r;
            im.data[i + y2 * im.w + 0 * im.w * im.h] = r;

            im.data[i + y1 * im.w + 1 * im.w * im.h] = g;
            im.data[i + y2 * im.w + 1 * im.w * im.h] = g;

            im.data[i + y1 * im.w + 2 * im.w * im.h] = b;
            im.data[i + y2 * im.w + 2 * im.w * im.h] = b;
        }
        for (i = y1; i <= y2; ++i)
        {
            im.data[x1 + i * im.w + 0 * im.w * im.h] = r;
            im.data[x2 + i * im.w + 0 * im.w * im.h] = r;

            im.data[x1 + i * im.w + 1 * im.w * im.h] = g;
            im.data[x2 + i * im.w + 1 * im.w * im.h] = g;

            im.data[x1 + i * im.w + 2 * im.w * im.h] = b;
            im.data[x2 + i * im.w + 2 * im.w * im.h] = b;
        }
    }
}

static float get_pixelBychannel(image m, int x, int y, int c)
{
    if (x < 0 || x >= m.w || y < 0 || y >= m.h || c < 0 || c >= m.c)
        return 0;
    return get_pixelData(m, x, y, c);
}

image copy_image(image p)
{
    image copy = p;
    copy.data = (float*)calloc((size_t)p.h * p.w * p.c, sizeof(float));
    memcpy(copy.data, p.data, (unsigned long)p.h * p.w * p.c * sizeof(float));
    return copy;
}

static void set_pixelData(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c)
        return;
    if (x < m.w && y < m.h && c < m.c)
    {
        m.data[c * m.h * m.w + y * m.w + x] = val;
    }
}

void add_image(image source, image dest, int dx, int dy)
{
    for (int k = 0; k < source.c; ++k)
    {
        for (int y = 0; y < source.h; ++y)
        {
            for (int x = 0; x < source.w; ++x)
            {
                int srcIndex = k * source.w * source.h + y * source.w + x;
                int dstIndex = k * dest.h * dest.w + (dy + y) * dest.w + (dx + x);
                dest.data[dstIndex] = source.data[srcIndex];
            }
        }
    }
}

void combination_image(image source, image dest, int dx, int dy)
{
    for (int k = 0; k < source.c; ++k)
    {
        for (int y = 0; y < source.h; ++y)
        {
            for (int x = 0; x < source.w; ++x)
            {
                int srcIndex = k * source.w * source.h + y * source.w + x;
                int dstIndex = k * dest.h * dest.w + (dy + y) * dest.w + (dx + x);
                dest.data[dstIndex] = source.data[srcIndex] * dest.data[dstIndex];
            }
        }
    }
}

image imread(const char* filename)
{
    return load_image_stb(filename, 0);
}

image imread2post(const char* filename)
{
    image im = load_image_stb(filename, 0);
    const int len = im.c * im.h * im.w;
    for (int i = 0; i < len; ++i)
    {
        im.data[i] *= 255;
    }
    return im;
}

image rgb2bgr_permute(image src)
{
    const int len = src.c * src.h * src.w;
    float* GRB = (float*)malloc(sizeof(float) * len);
    for (int c = 0; c < src.c; c++)
    {
        for (int h = 0; h < src.h; h++)
        {
            for (int w = 0; w < src.w; w++)
            {
                int newIndex = (c)*src.h * src.w + h * src.w + w;
                int grbIndex = (2 - c) * src.h * src.w + h * src.w + w;
                GRB[grbIndex] = src.data[newIndex];
            }
        }
    }
    for (int i = 0; i < len; ++i)
    {
        src.data[i] = GRB[i];
    }
    free(GRB);
    return src;
}

image image_permute(image src)
{
    float* GRB = (float*)malloc(sizeof(float) * src.c * src.h * src.w);
    for (int c = 0; c < src.c; c++)
    {
        for (int h = 0; h < src.h; h++)
        {
            for (int w = 0; w < src.w; w++)
            {
                int newIndex = (c)*src.h * src.w + h * src.w + w;
                int grbIndex = (2 - c) * src.h * src.w + h * src.w + w;
                GRB[grbIndex] = src.data[newIndex];
            }
        }
    }
    free(GRB);
    return src;
}

image gray2bgr(image src)
{
    image res;
    res.c = 3;
    res.h = src.h;
    res.w = src.w;
    res.data = (float*)malloc(sizeof(float) * 3 * src.h * src.w);
    for (int x = 0; x < src.h; x++)
    {
        for (int y = 0; y < src.w; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                res.data[x * src.w * src.c + y * 3 + i] = src.data[x * src.w + y];
            }
        }
    }
    free_image(src);
    return res;
}

image tranpose(image src)
{
    int size = src.c * src.h * src.w;
    float* tempData = (float*)malloc(sizeof(float) * size);
    int index = 0;

    for (int c = 0; c < src.c; c++)
    {
        for (int w = 0; w < src.w; w++)
        {
            for (int h = 0; h < src.h; h++)
            {
                tempData[index] = src.data[c * src.h * src.w + h * src.w + w];
                index++;
            }
        }
    }
    int tempH = src.h;
    int tempW = src.w;
    src.h = tempW;
    src.w = tempH;

    index = 0;
    for (int c = 0; c < src.c; c++)
    {
        for (int w = 0; w < tempH; w++)
        {
            for (int h = 0; h < tempW; h++)
            {
                src.data[c * src.h * src.w + w * tempW + h] = tempData[index];
                index++;
            }
        }
    }

    free(tempData);
    return src;
}

void draw_circle(image im, int x, int y, int radius, int r, int g, int b)
{
    int startX = x - radius;
    int startY = y - radius;
    int endX = x + radius;
    int endY = y + radius;
    if (startX < 0)
        startX = 0;
    if (startY < 0)
        startY = 0;
    if (endX > im.w)
        endX = im.w;
    if (endY > im.h)
        endY = im.h;

    for (int j = startY; j < endY; j++)
    {
        for (int i = startX; i < endX; i++)
        {
            int num1 = (i - x) * (i - x) + (j - y) * (j - y);
            int num2 = radius * radius;
            if (num1 <= num2)
            {
                im.data[0 * im.h * im.w + j * im.w + i] = r;
                im.data[1 * im.h * im.w + j * im.w + i] = g;
                im.data[2 * im.h * im.w + j * im.w + i] = b;
            }
        }
    }
}

void subtract(image a, image b, image c)
{
    const int size = a.c * a.h * a.w;
    for (int i = 0; i < size; i++)
    {
        c.data[i] = a.data[i] - b.data[i];
    }
    c.c = a.c;
    c.h = a.h;
    c.w = a.w;
}

void multi(image a, float value, image b)
{
    const int size = a.c * a.h * a.w;
    for (int i = 0; i < size; i++)
    {
        b.data[i] = a.data[i] * value;
    }
    b.c = a.c;
    b.h = a.h;
    b.w = a.w;
}

image rgb2gray(image src)
{
    image res;
    res.h = src.h;
    res.w = src.w;
    res.c = 1;
    res.data = (float*)malloc(sizeof(float) * res.h * res.w);
    for (int i = 0; i < res.h; i++)
    {
        for (int j = 0; j < res.w; j++)
        {
            float r = src.data[0 * src.h * src.w + i * src.w + j];
            float g = src.data[1 * src.h * src.w + i * src.w + j];
            float b = src.data[2 * src.h * src.w + i * src.w + j];
            res.data[i * res.w + j] = (r * 299 + g * 587 + b * 114 + 500) / 1000;
        }
    }
    free_image(src);
    return res;
}

void free_image(image m)
{
    if (m.data)
    {
        free(m.data);
    }
}

image letterbox(image im, int w, int h)
{
    int ow = im.w;
    int oh = im.h;
    if (((float)w / im.w) < ((float)h / im.h))
    {
        ow = w;
        oh = (im.h * w) / im.w;
    }
    else
    {
        oh = h;
        ow = (im.w * h) / im.h;
    }
    image resized = resize_image(im, ow, oh);
    image boxed;
    boxed.w = w;
    boxed.h = h;
    boxed.c = im.c;
    boxed.data = (float*)malloc(sizeof(float) * im.c * h * w);

    for (int i = 0; i < boxed.c * boxed.h * boxed.w; i++)
    {
        boxed.data[i] = 0.5;
    }

    add_image(resized, boxed, (w - ow) / 2, (h - oh) / 2);
    free_image(resized);

    return boxed;
}

void tengine_resize_f32(float* data, float* res, int ow, int oh, int c, int h, int w)
{
    float _scale_x = (float)(w) / (float)(ow);
    float _scale_y = (float)(h) / (float)(oh);
    float offset = 0.5f;

    int16_t* buf = (int16_t*)malloc((ow + ow + ow + oh + oh + oh) * sizeof(int16_t));
    int16_t* xCoef = (int16_t*)(buf);
    int16_t* xPos = (int16_t*)(buf + ow + ow);
    int16_t* yCoef = (int16_t*)(buf + ow + ow + ow);
    int16_t* yPos = (int16_t*)(buf + ow + ow + ow + oh + oh);

    for (int i = 0; i < ow; i++)
    {
        float fx = (float)(((float)i + offset) * _scale_x - offset);
        int sx = (int)fx;
        fx -= sx;
        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 0.f;
        }
        xCoef[i] = fx * 2048;
        xCoef[i + ow] = (1.f - fx) * 2048;
        xPos[i] = sx;
    }

    for (int j = 0; j < oh; j++)
    {
        float fy = (float)(((float)j + offset) * _scale_y - offset);
        int sy = (int)fy;
        fy -= sy;
        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= h - 1)
        {
            sy = h - 2;
            fy = 0.f;
        }
        yCoef[j] = fy * 2048;
        yCoef[j + oh] = (1.f - fy) * 2048;
        yPos[j] = sy;
    }

    //    int32_t* row = new int32_t[ow + ow];
    int32_t* row = (int32_t*)malloc((ow + ow) * sizeof(int32_t));

    for (int k = 0; k < c; k++)
    {
        int32_t channel = k * w * h;
        for (int j = 0; j < oh; j++)
        {
#ifdef __ARM_NEON
            int32x4_t fy_0 = vdupq_n_s32(yCoef[j + oh]);
            int32x4_t _fy = vdupq_n_s32(yCoef[j]);
#endif
            int32_t* p0_u = row;
            int32_t* p0_d = row + ow;
            int32_t yPosValue = yPos[j] * w + channel;
            for (int i = 0; i < ow; i++)
            {
                int32_t data0 = (int32_t) * (data + yPosValue + xPos[i]) * xCoef[i + ow] >> 11;
                int32_t data1 = (int32_t) * (data + yPosValue + xPos[i] + 1) * xCoef[i] >> 11;
                int32_t data2 = (int32_t) * (data + yPosValue + w + xPos[i]) * xCoef[i + ow] >> 11;
                int32_t data3 = (int32_t) * (data + yPosValue + w + xPos[i] + 1) * xCoef[i] >> 11;
                p0_u[i] = ((data0) + (data1));
                p0_d[i] = ((data2) + (data3));
            }
#ifdef __ARM_NEON
            for (int i = 0; i < (ow & -4); i += 4)
            {
                int32x4_t c1DataR = vmulq_s32(vld1q_s32(p0_u + i), fy_0);
                int32x4_t c1DataL = vmulq_s32(vld1q_s32(p0_d + i), _fy);
                int32x4_t c1Data_int = vshrq_n_s32(vaddq_s32(c1DataR, c1DataL), 11);
                float32x4_t c1Data_float = vcvtq_f32_s32(c1Data_int);
                vst1q_f32(res, c1Data_float);

                res += 4;
            }

            for (int i = ow & ~3; i < ow; i++)
            {
                int32_t data0 = *(p0_u + i) * yCoef[j + oh];
                int32_t data1 = *(p0_d + i) * yCoef[j];
                *res = (data0 + data1) >> 11;
                res++;
            }
#else
            for (int i = 0; i < ow; i++)
            {
                int32_t data0 = *(p0_u + i) * yCoef[j + oh];
                int32_t data1 = *(p0_d + i) * yCoef[j];
                *res = (data0 + data1) >> 11;
                res++;
            }
#endif
        }
    }

    free(row);
    free(buf);
}

void get_input_data(const char* image_file, float* input_data, int img_h, int img_w, const float* mean,
                    const float* scale)
{
    float means[3] = {mean[0], mean[1], mean[2]};
    float scales[3] = {scale[0], scale[1], scale[2]};
    image img = imread_process(image_file, img_w, img_h, means, scales);
    memcpy(input_data, img.data, sizeof(float) * img.c * img_w * img_h);
    free_image(img);
}

static void sort_cls_score(cls_score* array, int left, int right)
{
    int i = left;
    int j = right;
    cls_score key;

    if (left >= right)
        return;

    memmove(&key, &array[left], sizeof(cls_score));
    while (left < right)
    {
        while (left < right && key.score >= array[right].score)
        {
            --right;
        }
        memmove(&array[left], &array[right], sizeof(cls_score));
        while (left < right && key.score <= array[left].score)
        {
            ++left;
        }
        memmove(&array[right], &array[left], sizeof(cls_score));
    }

    memmove(&array[left], &key, sizeof(cls_score));

    sort_cls_score(array, i, left - 1);
    sort_cls_score(array, left + 1, j);
}

void print_topk(float* data, int total_num, int topk)
{
    cls_score* cls_scores = (cls_score*)malloc(total_num * sizeof(cls_score));
    for (int i = 0; i < total_num; i++)
    {
        cls_scores[i].id = i;
        cls_scores[i].score = data[i];
    }

    sort_cls_score(cls_scores, 0, total_num - 1);
    char strline[128] = "";
    for (int i = 0; i < topk; i++)
    {
        fprintf(stderr, "%f, %d\n", cls_scores[i].score, cls_scores[i].id);
    }
    free(cls_scores);
}
