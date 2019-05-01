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
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/roi_pooling.hpp"
namespace TEngine {

namespace ROIPoolingImpl {

struct ROIPoolingOps : public NodeOps
{
    bool Run(Node* node)
    {
        Tensor* feat_tensor = node->GetInputTensor(0);
        Tensor* roi_tensor = node->GetInputTensor(1);
        Tensor* output_tensor = node->GetOutputTensor(0);

        TShape& roi_shape = roi_tensor->GetShape();
        TShape& out_shape = output_tensor->GetShape();

        const float* featmap = ( float* )get_tensor_mem(feat_tensor);
        float* roi = ( float* )get_tensor_mem(roi_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);

        const std::vector<int>& dims = feat_tensor->GetShape().GetDim();

        const int channel = dims[1];
        const int height = dims[2];
        const int width = dims[3];
        const int feat_size = height * width;
        ROIPooling* roipooling_op = dynamic_cast<ROIPooling*>(node->GetOp());
        ROIPoolingParam* param_ = roipooling_op->GetParam();
        float spatial_scale = param_->spatial_scale;
        int pool_h = param_->pooled_h;
        int pool_w = param_->pooled_w;
        int pool_hw = pool_h * pool_w;

        const int num_roi = roi_shape.GetC();
        std::vector<int> outdim = {num_roi, channel, pool_h, pool_w};
        out_shape.SetDim(outdim);
        float* out_ptr = output;
        for(int i = 0; i < num_roi; i++)
        {
            const float* roi_ptr = roi + i * 4;
            int roi_x0 = round(roi_ptr[0] * spatial_scale);
            int roi_y0 = round(roi_ptr[1] * spatial_scale);
            int roi_x1 = round(roi_ptr[2] * spatial_scale);
            int roi_y1 = round(roi_ptr[3] * spatial_scale);
            int roi_w = std::max(roi_x1 - roi_x0 + 1, 1);
            int roi_h = std::max(roi_y1 - roi_y0 + 1, 1);
            float bin_w = ( float )roi_w / ( float )pool_w;
            float bin_h = ( float )roi_h / ( float )pool_h;
            for(int c = 0; c < channel; c++)
            {
                const float* feat_ptr = featmap + c * feat_size;
                for(int h = 0; h < pool_h; h++)
                {
                    for(int w = 0; w < pool_w; w++)
                    {
                        // h0: h_start
                        // h1: h_end
                        int h0 = roi_y0 + ( int )floor(( float )( h )*bin_h);
                        int h1 = roi_y0 + ( int )ceil(( float )(h + 1) * bin_h);
                        int w0 = roi_x0 + ( int )floor(( float )( w )*bin_w);
                        int w1 = roi_x0 + ( int )ceil(( float )(w + 1) * bin_w);
                        h0 = std::min(std::max(h0, 0), height);
                        h1 = std::min(std::max(h1, 0), height);
                        w0 = std::min(std::max(w0, 0), width);
                        w1 = std::min(std::max(w1, 0), width);
                        bool is_empty = (h1 <= h0) || (w1 <= w0);

                        float max_value = is_empty ? 0.f : feat_ptr[h0 * width + w0];
                        for(int y = h0; y < h1; y++)
                        {
                            for(int x = w0; x < w1; x++)
                            {
                                int idx = y * width + x;
                                max_value = std::max(max_value, feat_ptr[idx]);
                            }
                        }
                        out_ptr[h * pool_w + w] = max_value;
                    }
                }
                out_ptr += pool_hw;
            }
        }
        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    ROIPoolingOps* ops = new ROIPoolingOps();

    return ops;
}

}    // namespace ROIPoolingImpl

using namespace ROIPoolingImpl;

void RegisterROIPoolingNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "ROIPooling", ROIPoolingImpl::SelectFunc, 1000);
}

}    // namespace TEngine
