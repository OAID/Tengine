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
#include "operator/priorbox.hpp"
#include <math.h>
namespace TEngine {

namespace PriorBoxImpl {

struct PriorBoxOps : public NodeOps
{
    bool Run(Node* node)
    {
        const Tensor* data_tensor = node->GetInputTensor(1);
        const Tensor* featmap_tensor = node->GetInputTensor(0);

        Tensor* output_tensor = node->GetOutputTensor(0);

        float* output = ( float* )get_tensor_mem(output_tensor);

        PriorBox* priorbox_op = dynamic_cast<PriorBox*>(node->GetOp());
        PriorBoxParam* param_ = priorbox_op->GetParam();

        const TShape& data_shape = data_tensor->GetShape();
        const int data_height = data_shape.GetH();
        const int data_width = data_shape.GetW();
        const TShape& featmap_shape = featmap_tensor->GetShape();
        const int feat_height = featmap_shape.GetH();
        const int feat_width = featmap_shape.GetW();
        int img_w, img_h;
        if(param_->img_h == 0 || param_->img_w == 0)
        {
            img_w = data_width;
            img_h = data_height;
        }
        else
        {
            img_w = param_->img_w;
            img_h = param_->img_h;
        }
        float step_w, step_h;
        if(param_->step_h == 0 || param_->step_w == 0)
        {
            step_w = ( float )(img_w) / feat_width;
            step_h = ( float )(img_h) / feat_height;
        }
        else
        {
            step_w = param_->step_w;
            step_h = param_->step_h;
        }
        // out shape [feat_width,feat_height,num_priors_ * 4,2]
        int num_priors_ = param_->num_priors_;

        // default offset=0.5
        // box[xmin,ymin,xmax,ymax]
        float offset_ = param_->offset;
        for(int h = 0; h < feat_height; ++h)
        {
            float* box = output + h * num_priors_ * 4 * feat_width;
            for(int w = 0; w < feat_width; ++w)
            {
                float center_x = (w + offset_) * step_w;
                float center_y = (h + offset_) * step_h;
                float box_width, box_height;
                for(int s = 0; s < ( int )param_->min_size.size(); ++s)
                {
                    int min_size_ = param_->min_size[s];
                    // first prior: aspect_ratio = 1, size = min_size
                    box_width = box_height = min_size_;
                    box[0] = (center_x - box_width * 0.5f) / img_w;
                    box[1] = (center_y - box_height * 0.5f) / img_h;
                    box[2] = (center_x + box_width * 0.5f) / img_w;
                    box[3] = (center_y + box_height * 0.5f) / img_h;
                    box += 4;

                    // default：len(max_size)=len(min_size)
                    if(param_->max_size.size() > 0)
                    {
                        int max_size_ = param_->max_size[s];
                        // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        box_width = box_height = sqrt(min_size_ * max_size_);
                        box[0] = (center_x - box_width * 0.5f) / img_w;
                        box[1] = (center_y - box_height * 0.5f) / img_h;
                        box[2] = (center_x + box_width * 0.5f) / img_w;
                        box[3] = (center_y + box_height * 0.5f) / img_h;
                        box += 4;
                    }

                    // rest of priors
                    for(int r = 0; r < ( int )param_->aspect_ratio.size(); ++r)
                    {
                        float ar = param_->aspect_ratio[r];

                        box_width = min_size_ * sqrt(ar);
                        box_height = min_size_ / sqrt(ar);
                        box[0] = (center_x - box_width * 0.5f) / img_w;
                        box[1] = (center_y - box_height * 0.5f) / img_h;
                        box[2] = (center_x + box_width * 0.5f) / img_w;
                        box[3] = (center_y + box_height * 0.5f) / img_h;
                        box += 4;
                        if(param_->flip)
                        {
                            box[0] = (center_x - box_height * 0.5f) / img_h;
                            box[1] = (center_y - box_width * 0.5f) / img_w;
                            box[2] = (center_x + box_height * 0.5f) / img_h;
                            box[3] = (center_y + box_width * 0.5f) / img_w;
                            box += 4;
                        }
                    }
                }
            }
        }
        // clip the prior's coordidate such that it is within [0, 1]
        int dim = param_->out_dim_;
        if(param_->clip)
        {
            for(int d = 0; d < dim; ++d)
            {
                output[d] = std::min(std::max(output[d], 0.f), 1.f);
            }
        }
        // set the variance.
        float* output_ptr = output + dim;
        int size = dim / 4;
        for(int i = 0; i < size; i++)
        {
            output_ptr[0] = param_->variance[0];
            output_ptr[1] = param_->variance[1];
            output_ptr[2] = param_->variance[2];
            output_ptr[3] = param_->variance[3];
            output_ptr += 4;
        }

        return true;
    }
};

}    // namespace PriorBoxImpl

using namespace PriorBoxImpl;

void RegisterPriorBoxNodeExec(void)
{
    PriorBoxOps* ops = new PriorBoxOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "PriorBox", ops);
}

}    // namespace TEngine
