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
 * Author: haoluo@openailab.com
 */

#ifndef __REF_RPN_KERNEL_H__
#define __REF_RPN_KERNEL_H__

#include <stdint.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct anchor_box
{
    float x0;    // xmin
    float y0;    // ymin
    float x1;    // xmax
    float y1;    // ymax
};
struct RPN_Box
{
    float x0;    // xmin
    float y0;    // ymin
    float x1;    // xmax
    float y1;    // ymax
    float score;
};

struct rpn_param
{
    int feat_height;
    int feat_width;
    int feat_chan;
    int score_chan;
    float src_scale;
    int src_width;
    int src_height;
    int num_anchors;
    int min_size;
    int feat_stride;
    int per_nms_topn;
    int post_nms_topn;
    float nms_thresh;
    // float scales[4];
    // float quant_scale[3];
    // int zero[3];
};

#define RPN_MIN(a, b) ((a) < (b) ? (a) : (b))
#define RPN_MAX(a, b) ((a) > (b) ? (a) : (b))

typedef int (*ref_rpn_kernel_t)(const void* score, void* featmap, float* anchor, void* output, struct rpn_param* param);

static inline void bbox_tranform_inv(float* m_box, float* local_anchors, struct rpn_param* param)
{
    int feat_size = param->feat_height * param->feat_width;
    int c_4 = param->feat_chan / 4;
    for(int i = 0; i < c_4; ++i)
    {
        for(int j = 0; j < (2 * feat_size); ++j)
        {
            local_anchors[(i * 4 + 2) * feat_size + j] -= local_anchors[(i * 4 + 0) * feat_size + j] - 1;
            local_anchors[(i * 4 + 0) * feat_size + j] += local_anchors[(i * 4 + 2) * feat_size + j] * 0.5;

            m_box[(i * 4 + 0) * feat_size + j] *= local_anchors[(i * 4 + 2) * feat_size + j];
            m_box[(i * 4 + 0) * feat_size + j] += local_anchors[(i * 4 + 0) * feat_size + j];

            m_box[(i * 4 + 2) * feat_size + j] = exp(m_box[(i * 4 + 2) * feat_size + j]);
            m_box[(i * 4 + 2) * feat_size + j] *= local_anchors[(i * 4 + 2) * feat_size + j];
        }
    }
}

static inline void ref_filter_boxes(struct RPN_Box* boxes, const float* featmap, const float* score, int* num_boxes,
                                    struct rpn_param* param)
{
    float local_minsize = param->min_size * param->src_scale;
    int c_4 = param->feat_chan / 4;
    int feat_size = param->feat_height * param->feat_width;

    int offset_w, offset_h, offset_x, offset_y, offset_s;

    int num = 0;
    for(int h = 0; h < param->feat_height; h++)
        for(int w = 0; w < param->feat_width; w++)
        {
            offset_x = h * param->feat_width + w;
            offset_y = offset_x + feat_size;
            offset_w = offset_y + feat_size;
            offset_h = offset_w + feat_size;
            offset_s = feat_size * param->num_anchors + offset_x;
            for(int c = 0; c < c_4; c++)
            {
                float width = featmap[offset_w];
                float height = featmap[offset_h];
                if((width >= local_minsize) & (height >= local_minsize))
                {
                    struct RPN_Box tmp;
                    tmp.x0 = featmap[offset_x] - 0.5 * width;
                    tmp.y0 = featmap[offset_y] - 0.5 * height;
                    tmp.x1 = featmap[offset_x] + 0.5 * width;
                    tmp.y1 = featmap[offset_y] + 0.5 * height;
                    tmp.x0 = RPN_MIN(RPN_MAX(tmp.x0, 0), param->src_width);
                    tmp.y0 = RPN_MIN(RPN_MAX(tmp.y0, 0), param->src_height);
                    tmp.x1 = RPN_MIN(RPN_MAX(tmp.x1, 0), param->src_width);
                    tmp.y1 = RPN_MIN(RPN_MAX(tmp.y1, 0), param->src_height);
                    tmp.score = score[offset_s];
                    memcpy(boxes + num, &tmp, sizeof(struct RPN_Box));
                    num++;
                }
                offset_x += 4 * feat_size;
                offset_y += 4 * feat_size;
                offset_w += 4 * feat_size;
                offset_h += 4 * feat_size;
                offset_s += feat_size;
            }
        }

    *num_boxes = num;
}

void sort_rpn_boxes_by_score(struct RPN_Box* boxes, int size)
{
    int i, j;
    for(i = 0; i < size - 1; i++)
    {
        int max_idx = i;
        for(j = i + 1; j < size; j++)
        {
            if(boxes[max_idx].score < boxes[j].score)
                max_idx = j;
        }
        if(i != max_idx)
        {
            struct RPN_Box tmp;
            memcpy(&tmp, boxes + i, sizeof(struct RPN_Box));
            memcpy(boxes + i, boxes + max_idx, sizeof(struct RPN_Box));
            memcpy(boxes + max_idx, &tmp, sizeof(struct RPN_Box));
        }
    }
}

void nms_rpn_boxes(struct RPN_Box* input_boxes, int* size, float nms_thresh)
{
    int input_size = *size;
    int output_size = 0;

    struct RPN_Box* output_boxes = ( struct RPN_Box* )malloc(sizeof(struct RPN_Box) * input_size);
    float* areas = ( float* )malloc(sizeof(float) * input_size);
    int* picked = ( int* )malloc(sizeof(int) * input_size);

    for(int i = 0; i < input_size; ++i)
    {
        areas[i] = (input_boxes[i].x1 - input_boxes[i].x0 + 1) * (input_boxes[i].y1 - input_boxes[i].y0 + 1);
    }
    for(int i = 0; i < input_size; ++i)
    {
        int keep = 1;
        for(int j = 0; j < output_size; j++)
        {
            float xx1 = RPN_MAX(input_boxes[i].x0, output_boxes[j].x0);
            float yy1 = RPN_MAX(input_boxes[i].y0, output_boxes[j].y0);
            float xx2 = RPN_MIN(input_boxes[i].x1, output_boxes[j].x1);
            float yy2 = RPN_MIN(input_boxes[i].y1, output_boxes[j].y1);
            float w = RPN_MAX(float(0), xx2 - xx1 + 1);
            float h = RPN_MAX(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[picked[j]] - inter);
            if(ovr >= nms_thresh)
            {
                keep = 0;
                break;
            }
        }
        if(keep)
        {
            memcpy(output_boxes + output_size, input_boxes + i, sizeof(struct RPN_Box));
            picked[output_size] = i;
            output_size++;
        }
    }
    memcpy(input_boxes, output_boxes, output_size * sizeof(struct RPN_Box));
    *size = output_size;
    free(picked);
    free(areas);
    free(output_boxes);
}

#ifdef CONFIG_KERNEL_FP32
#include "ref_rpn_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_rpn_fp16.c"
#endif
/*

#ifdef CONFIG_KERNEL_UINT8
#include "ref_rpn_uint8.c"
#endif
*/

#ifdef __cplusplus
}
#endif

#endif
