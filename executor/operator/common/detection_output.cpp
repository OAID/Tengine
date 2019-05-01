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
#include "operator/detection_output.hpp"
#include <math.h>
namespace TEngine {

namespace DetectionOutputImpl {

struct DetectionOutputOps : public NodeOps
{
    struct Box
    {
        float x0;
        float y0;
        float x1;
        float y1;
        int class_idx;
        float score;
    };

    void get_boxes(std::vector<Box>& boxes, int num_prior, float* loc_ptr, float* prior_ptr)
    {
        for(int i = 0; i < num_prior; i++)
        {
            float* loc = loc_ptr + i * 4;
            float* pbox = prior_ptr + i * 4;
            float* pvar = pbox + num_prior * 4;
            // center size
            // pbox [xmin,ymin,xmax,ymax]
            float pbox_w = pbox[2] - pbox[0];
            float pbox_h = pbox[3] - pbox[1];
            float pbox_cx = (pbox[0] + pbox[2]) * 0.5f;
            float pbox_cy = (pbox[1] + pbox[3]) * 0.5f;

            // loc []
            float bbox_cx = pvar[0] * loc[0] * pbox_w + pbox_cx;
            float bbox_cy = pvar[1] * loc[1] * pbox_h + pbox_cy;
            float bbox_w = pbox_w * exp(pvar[2] * loc[2]);
            float bbox_h = pbox_h * exp(pvar[3] * loc[3]);
            // bbox [xmin,ymin,xmax,ymax]
            boxes[i].x0 = bbox_cx - bbox_w * 0.5f;
            boxes[i].y0 = bbox_cy - bbox_h * 0.5f;
            boxes[i].x1 = bbox_cx + bbox_w * 0.5f;
            boxes[i].y1 = bbox_cy + bbox_h * 0.5f;
        }
    }
    static inline float intersection_area(const Box& a, const Box& b)
    {
        if(a.x0 > b.x1 || a.x1 < b.x0 || a.y0 > b.y1 || a.y1 < b.y0)
        {
            // no intersection
            return 0.f;
        }

        float inter_width = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
        float inter_height = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);

        return inter_width * inter_height;
    }
    void nms_sorted_bboxes(const std::vector<Box>& bboxes, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();
        const int n = bboxes.size();

        std::vector<float> areas(n);
        for(int i = 0; i < n; i++)
        {
            const Box& r = bboxes[i];

            float width = r.x1 - r.x0;
            float height = r.y1 - r.y0;

            areas[i] = width * height;
        }

        for(int i = 0; i < n; i++)
        {
            const Box& a = bboxes[i];

            int keep = 1;
            for(int j = 0; j < ( int )picked.size(); j++)
            {
                const Box& b = bboxes[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                //             float IoU = inter_area / union_area
                if(inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if(keep)
                picked.push_back(i);
        }
    }

    bool Run(Node* node)
    {
        const Tensor* loc_tensor = node->GetInputTensor(0);
        const Tensor* conf_tensor = node->GetInputTensor(1);
        const Tensor* priorbox_tensor = node->GetInputTensor(2);
        Tensor* output_tensor = node->GetOutputTensor(0);

        DetectionOutput* detect_op = dynamic_cast<DetectionOutput*>(node->GetOp());
        DetectionOutputParam* param_ = detect_op->GetParam();

        // location   [b,num_prior*4,1,1]
        float* location = ( float* )get_tensor_mem(loc_tensor);
        // confidence [b,num_prior*21,1,1]
        float* confidence = ( float* )get_tensor_mem(conf_tensor);
        // priorbox   [b,2,num_prior*4,1]
        float* priorbox = ( float* )get_tensor_mem(priorbox_tensor);

        const std::vector<int>& dims = priorbox_tensor->GetShape().GetDim();
        const int num_priorx4 = dims[2];
        const int num_prior = num_priorx4 / 4;
        const int num_classes = param_->num_classes;
        // const int batch=dims[0];

        // only support for batch=1

        // for(int b=0;b<batch;b++)
        //{
        int b = 0;
        float* loc_ptr = location + b * num_priorx4;
        float* conf_ptr = confidence + b * num_prior * num_classes;
        float* prior_ptr = priorbox + b * num_priorx4 * 2;

        std::vector<Box> boxes(num_prior);
        get_boxes(boxes, num_prior, loc_ptr, prior_ptr);

        std::vector<std::vector<Box>> all_class_bbox_rects;
        all_class_bbox_rects.resize(num_classes);
        // start from 1 to ignore background class
        for(int i = 1; i < num_classes; i++)
        {
            std::vector<Box> class_box;
            for(int j = 0; j < num_prior; j++)
            {
                float score = conf_ptr[j * num_classes + i];
                if(score > param_->confidence_threshold)
                {
                    boxes[j].score = score;
                    boxes[j].class_idx = i;
                    class_box.push_back(boxes[j]);
                }
            }
            // sort
            std::sort(class_box.begin(), class_box.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

            // keep nms_top_k
            if(param_->nms_top_k < ( int )class_box.size())
            {
                class_box.resize(param_->nms_top_k);
            }
            // apply nms
            std::vector<int> picked;
            nms_sorted_bboxes(class_box, picked, param_->nms_threshold);
            // select
            for(int j = 0; j < ( int )picked.size(); j++)
            {
                int z = picked[j];
                all_class_bbox_rects[i].push_back(class_box[z]);
            }
        }
        // gather all class
        std::vector<Box> bbox_rects;

        for(int i = 0; i < num_classes; i++)
        {
            const std::vector<Box>& class_bbox_rects = all_class_bbox_rects[i];
            bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        }

        // global sort inplace
        std::sort(bbox_rects.begin(), bbox_rects.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        // keep_top_k
        if(param_->keep_top_k < ( int )bbox_rects.size())
        {
            bbox_rects.resize(param_->keep_top_k);
        }

        // output     [b,num,6,1]
        int num_detected = bbox_rects.size();
        int total_size = num_detected * 6 * 4;
        // alloc mem
        void* mem_addr = mem_alloc(total_size);
        set_tensor_mem(output_tensor, mem_addr, total_size, mem_free);
        float* output = ( float* )get_tensor_mem(output_tensor);

        TShape& out_shape = output_tensor->GetShape();
        std::vector<int> outdim = {1, num_detected, 6, 1};
        out_shape.SetDim(outdim);

        for(int i = 0; i < num_detected; i++)
        {
            const Box& r = bbox_rects[i];
            float* outptr = output + i * 6;
            outptr[0] = r.class_idx;
            outptr[1] = r.score;
            outptr[2] = r.x0;
            outptr[3] = r.y0;
            outptr[4] = r.x1;
            outptr[5] = r.y1;
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

    DetectionOutputOps* ops = new DetectionOutputOps();

    return ops;
}

}    // namespace DetectionOutputImpl

using namespace DetectionOutputImpl;

void RegisterDetectionOutputNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "DetectionOutput", DetectionOutputImpl::SelectFunc, 1000);
}

}    // namespace TEngine
