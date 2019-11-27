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

#ifndef __REF_DPP_KERNEL_H__
#define __REF_DPP_KERNEL_H__

#include <stdint.h>

#include "compiler_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Dpp_Box
{
    float x0;    // xmin
    float y0;    // ymin
    float x1;    // xmax
    float y1;    // ymax
    int box_idx;
    int class_idx;
    float score;
};

struct dpp_param
{
    int max_detections;
    int max_classes_per_detection;
    float nms_score_threshold;
    float nms_iou_threshold;
    int num_classes;
    int num_boxes;
    float scales[4];
    float quant_scale[3];
    int zero[3];
};

#define DPP_MIN(a, b) (a < b ? a : b)
#define DPP_MAX(a, b) (a > b ? a : b)

typedef int (*ref_dpp_kernel_t)(const void* input, const void* score, const void* anchor, void* detect_num,
                                void* detect_class, void* detect_score, void* detect_boxes, dpp_param* param);

static inline float intersection_area(const struct Dpp_Box a, const struct Dpp_Box b)
{
    if(a.x0 > b.x1 || a.x1 < b.x0 || a.y0 > b.y1 || a.y1 < b.y0)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = DPP_MIN(a.x1, b.x1) - DPP_MAX(a.x0, b.x0);
    float inter_height = DPP_MIN(a.y1, b.y1) - DPP_MAX(a.y0, b.y0);

    return inter_width * inter_height;
}

static inline void nms_sorted_bboxes(const struct Dpp_Box* boxes, int boxes_size, int* picked, int* picked_size,
                                     float nms_threshold)
{
    float areas[boxes_size];
    int n_picked = 0;
    for(int i = 0; i < boxes_size; i++)
    {
        float width = boxes[i].x1 - boxes[i].x0;
        float height = boxes[i].y1 - boxes[i].y0;

        areas[i] = width * height;
    }

    for(int i = 0; i < boxes_size; i++)
    {
        int keep = 1;
        for(int j = 0; j < n_picked; j++)
        {
            // intersection over union
            float inter_area = intersection_area(boxes[i], boxes[picked[j]]);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if(keep)
        {
            picked[n_picked] = i;
            n_picked++;
        }
    }
    *picked_size = n_picked;
}

void sort_boxes_by_score(struct Dpp_Box* boxes, int size)
{
    int i, j;
    for(i = 0; i < size - 1; i++)
    {
        int max_idx = i;
        for(j = i + 1; j < size; j++)
        {
            if(boxes[j].score < 0.6)
                continue;
            if(boxes[max_idx].score < boxes[j].score)
                max_idx = j;
        }
        if(i != max_idx)
        {
            struct Dpp_Box tmp;
            memcpy(&tmp, boxes + i, sizeof(struct Dpp_Box));
            memcpy(boxes + i, boxes + max_idx, sizeof(struct Dpp_Box));
            memcpy(boxes + max_idx, &tmp, sizeof(struct Dpp_Box));
        }
        else
        {
            if(boxes[max_idx].score < 0.6)
                return;
        }
    }
}

static inline int decode_single_box(struct Dpp_Box* box, const float* box_ptr, const float* anchor_ptr,
                                    const float* scales)
{
    int i = box->box_idx;

    const float* box_coord = box_ptr + i * 4;
    const float* anchor = anchor_ptr + i * 4;

    // [0]: y  [1]: x  [2]: h  [3]: w
    float ycenter = box_coord[0] / scales[0] * anchor[2] + anchor[0];
    float xcenter = box_coord[1] / scales[1] * anchor[3] + anchor[1];
    float half_h = 0.5f * (exp(box_coord[2] / scales[2])) * anchor[2];
    float half_w = 0.5f * (exp(box_coord[3] / scales[3])) * anchor[3];

    box->y0 = ycenter - half_h;
    box->x0 = xcenter - half_w;
    box->y1 = ycenter + half_h;
    box->x1 = xcenter + half_w;
    if(box->y0 < 0 || box->x0 < 0)
        return -1;
    return 0;
}

void get_all_boxes_rect(struct Dpp_Box* all_class_bbox_rects, const float* box, const float* scores,
                        const float* anchor, int num_boxes, int num_classes, float* scales)
{
    struct Dpp_Box selected_box;
    for(int j = 0; j < num_boxes; j++)
    {
        for(int i = 1; i < num_classes; i++)
        {
            float score = scores[j * num_classes + i];

            if(score < 0.6)
                continue;

            selected_box.score = score;
            selected_box.class_idx = i;
            selected_box.box_idx = j;
            // printf("score: %f ,box_idx: %d ,class: %d\n",score, j, i);

            if(decode_single_box(&selected_box, box, anchor, scales) < 0)
                continue;

            // struct Box* cls_vector = all_class_bbox_rects[i];
            memcpy(all_class_bbox_rects + i * num_boxes + j, &selected_box, sizeof(struct Dpp_Box));
        }
    }
}

int ref_dpp_common(const float* input_f, const float* score_f, const float* anchor_f, dpp_param* param,
                   float* detect_num, float* detect_class, float* detect_score, float* detect_boxes)
{
    const int num_classes = param->num_classes + 1;
    const int num_boxes = param->num_boxes;
    const int max_detections = param->max_detections;

    struct Dpp_Box* all_boxes = ( struct Dpp_Box* )malloc(num_classes * num_boxes * sizeof(struct Dpp_Box));
    memset(all_boxes, 0, sizeof(struct Dpp_Box) * num_classes * num_boxes);

    get_all_boxes_rect(all_boxes, input_f, score_f, anchor_f, num_boxes, num_classes, param->scales);

    int max_picked_boxes = 2 * max_detections * num_classes;
    struct Dpp_Box* picked_boxes = ( struct Dpp_Box* )malloc(max_picked_boxes * sizeof(struct Dpp_Box));
    memset(picked_boxes, 0, sizeof(struct Dpp_Box) * max_picked_boxes);
    int all_picked_size = 0;

    for(int i = 1; i < num_classes; i++)
    {
        struct Dpp_Box* class_box = all_boxes + i * num_boxes;

        // sort
        sort_boxes_by_score(class_box, num_boxes);
        int box_size = 0;
        for(int j = 0; j < num_boxes; j++)
        {
            if(class_box[j].score < 0.6)
                break;
            box_size++;
        }
        if(box_size == 0)
            continue;

        if(box_size > max_detections * 2)
            box_size = max_detections * 2;

        int picked[num_boxes];
        int picked_size = 0;

        picked[0] = 0;
        nms_sorted_bboxes(class_box, box_size, picked, &picked_size, param->nms_iou_threshold);

        // save the survivors
        for(int j = 0; j < picked_size; j++)
        {
            int z = picked[j];
            memcpy(picked_boxes + all_picked_size, class_box + z, sizeof(struct Dpp_Box));
            all_picked_size++;
        }
    }

    sort_boxes_by_score(picked_boxes, max_picked_boxes);
    if(all_picked_size > max_detections)
        all_picked_size = max_detections;

    // generate output tensors
    detect_num[0] = all_picked_size;

    for(int i = 0; i < all_picked_size; i++)
    {
        detect_class[i] = picked_boxes[i].class_idx;
        detect_score[i] = picked_boxes[i].score;

        detect_boxes[4 * i] = picked_boxes[i].x0;
        detect_boxes[4 * i + 1] = picked_boxes[i].y0;
        detect_boxes[4 * i + 2] = picked_boxes[i].x1;
        detect_boxes[4 * i + 3] = picked_boxes[i].y1;
    }

    free(all_boxes);
    free(picked_boxes);

    return 0;
}

#ifdef CONFIG_KERNEL_FP32
#include "ref_dpp_fp32.c"
#endif

#ifdef CONFIG_KERNEL_FP16
#include "ref_dpp_fp16.c"
#endif

#ifdef CONFIG_KERNEL_UINT8
#include "ref_dpp_uint8.c"
#endif

#ifdef __cplusplus
}
#endif

#endif
