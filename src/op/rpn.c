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
 * Author: zpluo@openailab.com
 */

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "sys_port.h"
#include "tengine_ir.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_op.h"
#include "parameter.h"
#include "rpn_param.h"

DEFINE_PARM_PARSE_ENTRY(rpn_param, ratios, anchor_scales, feat_stride, basesize, min_size, per_nms_topn, post_nms_topn,
                        nms_thresh, anchors_);

void mkanchor(float w, float h, float x_ctr, float y_ctr, Anchor_t* tmp)
{
    tmp->x0 = (x_ctr - 0.5f * (w - 1));
    tmp->y0 = (y_ctr - 0.5f * (h - 1));
    tmp->x1 = (x_ctr + 0.5f * (w - 1));
    tmp->y1 = (y_ctr + 0.5f * (h - 1));
}

void whctrs(const Anchor_t anchor, Box_t* result)
{
    result->w = (anchor.x1 - anchor.x0 + 1);
    result->h = (anchor.y1 - anchor.y0 + 1);
    result->cx = ((anchor.x1 + anchor.x0) * 0.5f);
    result->cy = ((anchor.y1 + anchor.y0) * 0.5f);
}

void scale_enum(const Anchor_t anchor, const struct vector* anchor_scales_, struct vector* result)
{
    Box_t tmp_box;
    whctrs(anchor, &tmp_box);

    for (int i = 0; i < ( int )anchor_scales_->elem_num; ++i)
    {
        Anchor_t tmp;

        float as_val = *( float* )(get_vector_data(( struct vector* )anchor_scales_, i));
        mkanchor(tmp_box.w * as_val, tmp_box.h * as_val, tmp_box.cx, tmp_box.cy, &tmp);
        push_vector_data(result, &tmp);
    }
}

void ratio_enum(const Anchor_t anchor, const struct vector* ratios_, struct vector* result)
{
    Box_t tmp_box;
    whctrs(anchor, &tmp_box);
    float area = tmp_box.h * tmp_box.w;

    for (int i = 0; i < ( int )ratios_->elem_num; ++i)
    {
        float size_ratio = area / *( float* )(get_vector_data(( struct vector* )ratios_, i));
        Anchor_t tmp;
        float new_w = round(sqrt(size_ratio));
        float new_h = round((double )new_w * *( float* )(get_vector_data(( struct vector* )ratios_, i)));
        mkanchor(new_w, new_h, tmp_box.cx, tmp_box.cy, &tmp);
        push_vector_data(result, &tmp);
    }
}

void generate_anchors(const int base_size, const struct vector* ratios_, const struct vector* scales_,
                      struct vector* gen_anchors_)
{
    Anchor_t base_anchor;
    base_anchor.x0 = 0.f;
    base_anchor.y0 = 0.f;
    base_anchor.x1 = base_size - 1.f;
    base_anchor.y1 = base_size - 1.f;

    struct vector* ratio_anchors = create_vector(sizeof(struct Anchor), NULL);

    ratio_enum(base_anchor, ratios_, ratio_anchors);
    for (int i = 0; i < ( int )ratio_anchors->elem_num; ++i)
    {
        struct vector* scale_anchors = create_vector(sizeof(struct Anchor), NULL);

        scale_enum(*( Anchor_t* )get_vector_data(ratio_anchors, i), scales_, scale_anchors);
        for (int j = 0; j < scale_anchors->elem_num; j++)
        {
            Anchor_t tmp_s = *( Anchor_t* )get_vector_data(scale_anchors, j);
            push_vector_data(gen_anchors_, &tmp_s);
        }

        release_vector(scale_anchors);
    }

    release_vector(ratio_anchors);
}

static int infer_shape(struct ir_node* node)
{
    struct ir_graph* ir_graph = node->graph;
    struct ir_tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct ir_tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    rpn_param_t* rpn_param = ( rpn_param_t* )node->op.param_mem;

    rpn_param->anchors_ = create_vector(sizeof(struct Anchor), NULL);
    generate_anchors(rpn_param->basesize, rpn_param->ratios, rpn_param->anchor_scales, rpn_param->anchors_);

    int dims[4];
    dims[0] = input->dims[0];
    dims[1] = rpn_param->post_nms_topn + 1;
    dims[2] = 4;
    dims[3] = 1;

    set_ir_tensor_shape(output, dims, 4);
    return 0;
}

static int init_op(struct ir_op* op)
{
    struct rpn_param* rpn_param = ( struct rpn_param* )sys_malloc(sizeof(struct rpn_param));

    if (rpn_param == NULL)
    {
        set_tengine_errno(ENOMEM);
        return -1;
    }

    /*set the param default value */
    rpn_param->feat_stride = 16;
    rpn_param->ratios = NULL;
    rpn_param->anchors_ = NULL;
    rpn_param->anchor_scales = NULL;

    op->param_mem = rpn_param;
    op->param_size = sizeof(struct rpn_param);
    op->same_shape = 0;
    op->infer_shape = infer_shape;

    return 0;
}

static void release_op(struct ir_op* op)
{
    struct rpn_param* rpn_param = ( struct rpn_param* )op->param_mem;

    if (rpn_param->anchors_)
        release_vector(rpn_param->anchors_);
    if (rpn_param->anchor_scales)
        release_vector(rpn_param->anchor_scales);
    if (rpn_param->ratios)
        release_vector(rpn_param->ratios);

    sys_free(op->param_mem);
}

static int register_rpn_op(void* arg)
{
    struct op_method m;

    m.op_version = 1;
    m.init_op = init_op;
    m.release_op = release_op;
    m.access_param_entry = access_param_entry;

    return register_op(OP_RPN, OP_RPN_NAME, &m);
}

static int unregister_rpn_op(void* arg)
{
    sys_free(GET_PARAM_PARSE_MAP(rpn_param));
    return unregister_op(OP_RPN, 1);
}

AUTO_REGISTER_OP(register_rpn_op);
AUTO_UNREGISTER_OP(unregister_rpn_op);
