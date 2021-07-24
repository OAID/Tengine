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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: zpluo@openailab.com
 */

#ifndef __RPN_PARAM_H__
#define __RPN_PARAM_H__

typedef struct Anchor
{
    float x0;
    float y0;
    float x1;
    float y1;
} Anchor_t;

typedef struct Box
{
    float w;
    float h;
    float cx;
    float cy;
} Box_t;

typedef struct rpn_param
{
    struct vector* ratios;
    struct vector* anchor_scales;
    int feat_stride;
    int basesize;
    int min_size;
    int per_nms_topn;
    int post_nms_topn;
    float nms_thresh;
    struct vector* anchors_;
} rpn_param_t;

#endif
