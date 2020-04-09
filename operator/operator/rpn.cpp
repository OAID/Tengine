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
#include "operator/rpn.hpp"
#include <math.h>

void mkanchor(float w, float h, float x_ctr, float y_ctr, Anchor* tmp)
{
    tmp->x0 = (x_ctr - 0.5 * (w - 1));
    tmp->y0 = (y_ctr - 0.5 * (h - 1));
    tmp->x1 = (x_ctr + 0.5 * (w - 1));
    tmp->y1 = (y_ctr + 0.5 * (h - 1));
}

void whctrs(const Anchor anchor, Box* result)
{
    result->w = (anchor.x1 - anchor.x0 + 1);
    result->h = (anchor.y1 - anchor.y0 + 1);
    result->cx = ((anchor.x1 + anchor.x0) * 0.5f);
    result->cy = ((anchor.y1 + anchor.y0) * 0.5f);
}

void scale_enum(const Anchor anchor, const std::vector<float>& anchor_scales_, std::vector<Anchor>& result)
{
    Box tmp_box;
    whctrs(anchor, &tmp_box);
    for(int i = 0; i < ( int )anchor_scales_.size(); ++i)
    {
        Anchor tmp;
        mkanchor(tmp_box.w * anchor_scales_[i], tmp_box.h * anchor_scales_[i], tmp_box.cx, tmp_box.cy, &tmp);
        result.push_back(tmp);
    }
}

void ratio_enum(const Anchor anchor, const std::vector<float>& ratios_, std::vector<Anchor>& result)
{
    Box tmp_box;
    whctrs(anchor, &tmp_box);
    float area = tmp_box.h * tmp_box.w;
    for(int i = 0; i < ( int )ratios_.size(); ++i)
    {
        float size_ratio = area / ratios_[i];
        Anchor tmp;
        float new_w = round(sqrt(size_ratio));
        float new_h = round(new_w * ratios_[i]);
        mkanchor(new_w, new_h, tmp_box.cx, tmp_box.cy, &tmp);
        result.push_back(tmp);
    }
}

void generate_anchors(const int base_size, const std::vector<float>& ratios_, const std::vector<float>& scales_,
                      std::vector<Anchor>& gen_anchors_)
{
    Anchor base_anchor;
    base_anchor.x0 = 0;
    base_anchor.y0 = 0;
    base_anchor.x1 = base_size - 1;
    base_anchor.y1 = base_size - 1;
    std::vector<Anchor> ratio_anchors;

    gen_anchors_.clear();

    ratio_enum(base_anchor, ratios_, ratio_anchors);
    for(int i = 0; i < ( int )ratio_anchors.size(); ++i)
    {
        std::vector<Anchor> scale_anchors;
        scale_enum(ratio_anchors[i], scales_, scale_anchors);
        gen_anchors_.insert(gen_anchors_.end(), scale_anchors.begin(), scale_anchors.end());
    }
}

namespace TEngine {

bool RPN::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape, int layout)
{
    generate_anchors(param_.basesize, param_.ratios, param_.anchor_scales, param_.anchors_);
    const TShape& input = ishape[0];
    const std::vector<int>& feat_dim = input.GetDim();

    TShape shape;
    std::vector<int> dim = {feat_dim[0], param_.post_nms_topn + 1, 4, 1};
    shape.SetDim(dim);
    shape.SetDataLayout(input.GetDataLayout());
    oshape[0] = shape;
    return true;
}

void RPN::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("feat_stride", 16)

        .SetDoc(R"DOC(RPN Layer)DOC");
}

}    // namespace TEngine
