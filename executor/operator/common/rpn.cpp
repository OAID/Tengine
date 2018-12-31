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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/rpn.hpp"
#include <math.h>

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
void _set(const int N, const float alpha, float* Y)
{
    for(int i = 0; i < N; ++i)
        Y[i] = alpha;
}

void _axpy(const int N, const float* X, float* Y)
{
    for(int i = 0; i < N; i++)
        Y[i] += X[i];
}
void _axpy_(const int N, const float* X, float* Y)
{
    for(int i = 0; i < N; i++)
        Y[i] -= X[i];
}
void _axpy_half(const int N, const float* X, float* Y)
{
    for(int i = 0; i < N; i++)
        Y[i] += 0.5 * X[i];
}
void _add_one(const int N, float* Y)
{
    for(int i = 0; i < N; i++)
        Y[i] += 1.f;
}
void _mul(const int N, const float* a, const float* b, float* y)
{
    for(int i = 0; i < N; i++)
        y[i] = a[i] * b[i];
}
void _add(const int N, const float* a, const float* b, float* y)
{
    for(int i = 0; i < N; i++)
        y[i] = a[i] + b[i];
}
void _exp(const int N, const float* a, float* y)
{
    for(int i = 0; i < N; i++)
        y[i] = exp(a[i]);
}

struct SBox
{
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
    bool operator<(const SBox& tmp) const
    {
        return score < tmp.score;
    }
};
void proposal_local_anchor(int feat_height, int feat_width, int feat_stride, std::vector<Anchor>& anchors,
                           float* local_anchors)
{
    int length = std::max(feat_height, feat_width);
    int feat_size = feat_height * feat_width;
    int* map_m = new int[length];
    for(int i = 0; i < length; ++i)
    {
        map_m[i] = i * feat_stride;
    }
    float* shift_x = new float[feat_size];
    float* shift_y = new float[feat_size];
    for(int i = 0; i < feat_height; ++i)
    {
        for(int j = 0; j < feat_width; ++j)
        {
            shift_x[i * feat_width + j] = map_m[j];
            shift_y[i * feat_width + j] = map_m[i];
        }
    }
    int num_anchors = ( int )anchors.size();
    float* a = local_anchors;
    for(int i = 0; i < num_anchors; ++i)
    {
        _set(feat_size, anchors[i].x0, a + (i * 4 + 0) * feat_size);
        _set(feat_size, anchors[i].y0, a + (i * 4 + 1) * feat_size);
        _set(feat_size, anchors[i].x1, a + (i * 4 + 2) * feat_size);
        _set(feat_size, anchors[i].y1, a + (i * 4 + 3) * feat_size);
        _axpy(feat_size, shift_x, a + (i * 4 + 0) * feat_size);
        _axpy(feat_size, shift_x, a + (i * 4 + 2) * feat_size);
        _axpy(feat_size, shift_y, a + (i * 4 + 1) * feat_size);
        _axpy(feat_size, shift_y, a + (i * 4 + 3) * feat_size);
    }
    delete[] map_m;
    delete[] shift_x;
    delete[] shift_y;
}

void bbox_tranform_inv(float* m_box, float* local_anchors, int height, int width, int channel, int num_anchors)
{
    int step = height * width;
    float* a = m_box;
    float* b = local_anchors;
    int c_4 = channel / 4;
    for(int i = 0; i < c_4; ++i)
    {
        _axpy_(2 * step, b + (i * 4 + 0) * step, b + (i * 4 + 2) * step);
        _add_one(2 * step, b + (i * 4 + 2) * step);
        _axpy_half(2 * step, b + (i * 4 + 2) * step, b + (i * 4 + 0) * step);

        _mul(2 * step, b + (i * 4 + 2) * step, a + (i * 4 + 0) * step, a + (i * 4 + 0) * step);
        _add(2 * step, b + (i * 4 + 0) * step, a + (i * 4 + 0) * step, a + (i * 4 + 0) * step);

        _exp(2 * step, a + (i * 4 + 2) * step, a + (i * 4 + 2) * step);
        _mul(2 * step, b + (i * 4 + 2) * step, a + (i * 4 + 2) * step, a + (i * 4 + 2) * step);
    }
}

void filter_boxs(std::vector<SBox>& boxes, float* box, float* score, int min_size, int src_scale, int src_w, int src_h,
                 int feat_w, int feat_h, int num_anchors, int feat_c)
{
    float local_minsize = min_size * src_scale;
    boxes.clear();

    int feat_c_ = feat_c / 4;
    int one_step = feat_h * feat_w;
    int step = 4 * one_step;
    int offset_w, offset_h, offset_x, offset_y, offset_s;

    for(int h = 0; h < feat_h; ++h)
    {
        for(int w = 0; w < feat_w; ++w)
        {
            offset_x = h * feat_w + w;
            offset_y = offset_x + one_step;
            offset_w = offset_y + one_step;
            offset_h = offset_w + one_step;
            offset_s = one_step * num_anchors + offset_x;
            for(int c = 0; c < feat_c_; ++c)
            {
                float width = box[offset_w];
                float height = box[offset_h];
                if((width >= local_minsize) & (height >= local_minsize))
                {
                    SBox tmp;
                    tmp.x0 = box[offset_x] - 0.5 * width;
                    tmp.y0 = box[offset_y] - 0.5 * height;
                    tmp.x1 = box[offset_x] + 0.5 * width;
                    tmp.y1 = box[offset_y] + 0.5 * height;
                    tmp.x0 = MIN(MAX(tmp.x0, 0), src_w);
                    tmp.y0 = MIN(MAX(tmp.y0, 0), src_h);
                    tmp.x1 = MIN(MAX(tmp.x1, 0), src_w);
                    tmp.y1 = MIN(MAX(tmp.y1, 0), src_h);
                    tmp.score = score[offset_s];
                    boxes.push_back(tmp);
                }
                offset_x += step;
                offset_y += step;
                offset_w += step;
                offset_h += step;
                offset_s += one_step;
            }
        }
    }
}

void nms_rpn(std::vector<SBox>& input_boxes, float nms_thresh)
{
    std::vector<float> vArea(input_boxes.size());
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        vArea[i] =
            (input_boxes.at(i).x1 - input_boxes.at(i).x0 + 1) * (input_boxes.at(i).y1 - input_boxes.at(i).y0 + 1);
    }
    for(int i = 0; i < ( int )input_boxes.size(); ++i)
    {
        for(int j = i + 1; j < ( int )input_boxes.size();)
        {
            float xx1 = std::max(input_boxes[i].x0, input_boxes[j].x0);
            float yy1 = std::max(input_boxes[i].y0, input_boxes[j].y0);
            float xx2 = std::min(input_boxes[i].x1, input_boxes[j].x1);
            float yy2 = std::min(input_boxes[i].y1, input_boxes[j].y1);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if(ovr >= nms_thresh)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
namespace TEngine {

namespace RPNImpl {

struct RPNOps : public NodeOps
{
    bool Run(Node* node)
    {
        const Tensor* score_tensor = node->GetInputTensor(0);
        const Tensor* featmap_tensor = node->GetInputTensor(1);
        const Tensor* info_tensor = node->GetInputTensor(2);
        Tensor* output_tensor = node->GetOutputTensor(0);
        TShape& out_shape = output_tensor->GetShape();

        float* output = ( float* )get_tensor_mem(output_tensor);
        const float* im_info = ( float* )get_tensor_mem(info_tensor);
        const float* m_score_ = ( float* )get_tensor_mem(score_tensor);
        const float* m_box_ = ( float* )get_tensor_mem(featmap_tensor);

        const TShape& featmap_shape = featmap_tensor->GetShape();
        const TShape& score_shape = score_tensor->GetShape();
        const int feat_height = featmap_shape.GetH();
        const int feat_width = featmap_shape.GetW();
        const int feat_channel = featmap_shape.GetC();
        const int score_channel = score_shape.GetC();
        const int feat_size = feat_height * feat_width;
        int src_height_ = im_info[0];
        int src_width_ = im_info[1];
        int src_scale_ = im_info[2];

        RPN* RPN_op = dynamic_cast<RPN*>(node->GetOp());
        RPNParam* param_ = RPN_op->GetParam();
        int feat_stride = param_->feat_stride;
        int num_anchors = ( int )param_->anchors_.size();

        // local_anchors (1, anchors_nums_ * 4, map_height_, map_width_);
        float* local_anchors = new float[num_anchors * 4 * feat_size];
        proposal_local_anchor(feat_height, feat_width, feat_stride, param_->anchors_, local_anchors);

        float* m_box = new float[feat_channel * feat_size];
        for(int i = 0; i < feat_channel * feat_size; i++)
            m_box[i] = m_box_[i];
        bbox_tranform_inv(m_box, local_anchors, feat_height, feat_width, feat_channel, num_anchors);

        delete[] local_anchors;
        std::vector<SBox> boxes;
        float* m_score = new float[score_channel * feat_size];
        for(int i = 0; i < score_channel * feat_size; i++)
            m_score[i] = m_score_[i];
        filter_boxs(boxes, m_box, m_score, param_->min_size, src_scale_, src_width_, src_height_, feat_width,
                    feat_height, num_anchors, feat_channel);
        delete[] m_box;
        delete[] m_score;

        std::sort(boxes.rbegin(), boxes.rend());

        if(param_->per_nms_topn > 0)
        {
            int tmp = MIN(param_->per_nms_topn, ( int )boxes.size());
            boxes.erase(boxes.begin() + tmp, boxes.end());
        }
        nms_rpn(boxes, param_->nms_thresh);

        if(param_->post_nms_topn > 0)
        {
            int tmp = MIN(param_->post_nms_topn, ( int )boxes.size());

            boxes.erase(boxes.begin() + tmp, boxes.end());
        }
        // inder shape [default batch=1]
        int num_box = boxes.size();
        std::vector<int> outdim = {1, num_box, 4, 1};
        out_shape.SetDim(outdim);

        float* out_data = output;
        // std::cout<<"num_box "<<num_box<<"\n";
        for(int i = 0; i < num_box; i++)
        {
            const SBox& r = boxes[i];
            float* outptr = out_data + i * 4;
            outptr[0] = r.x0;
            outptr[1] = r.y0;
            outptr[2] = r.x1;
            outptr[3] = r.y1;
        }
        return true;
    }
};

}    // namespace RPNImpl

using namespace RPNImpl;

void RegisterRPNNodeExec(void)
{
    RPNOps* ops = new RPNOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "RPN", ops);
}

}    // namespace TEngine
