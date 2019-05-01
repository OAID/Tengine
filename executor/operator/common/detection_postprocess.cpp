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
 * Author: jingyou@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <math.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/detection_postprocess.hpp"
#include "prof_utils.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace DetectionPostProcessImpl {

struct DetectionPostProcessOps : public NodeOps
{
    struct Box
    {
        float x0;    // xmin
        float y0;    // ymin
        float x1;    // xmax
        float y1;    // ymax
        int box_idx;
        int class_idx;
        float score;
    };

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

    static inline void nms_sorted_bboxes(const std::vector<Box>& bboxes, std::vector<int>& picked, float nms_threshold)
    {
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
                // float IoU = inter_area / union_area
                if(inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if(keep)
                picked.push_back(i);
        }
    }

    void decode_boxes(std::vector<Box>& decoded_boxes, const float* box_ptr, const float* anchor_ptr,
                      const DetectionPostProcessParam* param)
    {
        const int num_boxes = decoded_boxes.size();
        std::vector<float> scales = param->scales;

        for(int i = 0; i < num_boxes; i++)
        {
            const float* box_coord = box_ptr + i * 4;
            const float* anchor = anchor_ptr + i * 4;

            // [0]: y  [1]: x  [2]: h  [3]: w
            float ycenter = box_coord[0] / scales[0] * anchor[2] + anchor[0];
            float xcenter = box_coord[1] / scales[1] * anchor[3] + anchor[1];
            float half_h = 0.5f * static_cast<float>(std::exp(box_coord[2] / scales[2])) * anchor[2];
            float half_w = 0.5f * static_cast<float>(std::exp(box_coord[3] / scales[3])) * anchor[3];

            decoded_boxes[i].y0 = ycenter - half_h;
            decoded_boxes[i].x0 = xcenter - half_w;
            decoded_boxes[i].y1 = ycenter + half_h;
            decoded_boxes[i].x1 = xcenter + half_w;
        }
    }

    static inline int decode_single_box(Box& box, const float* box_ptr, const float* anchor_ptr,
                                        const std::vector<float>& scales)
    {
        int i = box.box_idx;

        const float* box_coord = box_ptr + i * 4;
        const float* anchor = anchor_ptr + i * 4;

        // [0]: y  [1]: x  [2]: h  [3]: w
        float ycenter = box_coord[0] / scales[0] * anchor[2] + anchor[0];
        float xcenter = box_coord[1] / scales[1] * anchor[3] + anchor[1];
        float half_h = 0.5f * static_cast<float>(std::exp(box_coord[2] / scales[2])) * anchor[2];
        float half_w = 0.5f * static_cast<float>(std::exp(box_coord[3] / scales[3])) * anchor[3];

        box.y0 = ycenter - half_h;
        box.x0 = xcenter - half_w;
        box.y1 = ycenter + half_h;
        box.x1 = xcenter + half_w;
        if(box.y0 < 0 || box.x0 < 0)
            return -1;
        return 0;
    }
    int box_zero = 0;
    int score_zero = 0;
    int anchor_zero = 0;

    template <typename type>
    void get_all_boxes_rect(std::vector<std::vector<Box>>& all_class_bbox_rects, uint8_t* box, uint8_t* score,
                            uint8_t* anchor, float box_scale, float score_scale, float anchor_scale, int num_boxes,
                            int num_classes, std::vector<float>& scales)
    {
        float* box_ptr = nullptr;
        float* score_ptr = nullptr;
        float* anchor_ptr = nullptr;
        if(sizeof(type) == 4)
        {
            box_ptr = ( float* )box;
            score_ptr = ( float* )score;
            anchor_ptr = ( float* )anchor;
        }
        else if(sizeof(type) == 1)
        {
            box_ptr = ( float* )std::malloc(sizeof(float) * num_boxes * 4);
            anchor_ptr = ( float* )std::malloc(sizeof(float) * num_boxes * 4);
            score_ptr = ( float* )std::malloc(sizeof(float) * num_boxes * (num_classes));
            for(int i = 0; i < num_boxes * 4; i++)
            {
                box_ptr[i] = (box[i] - box_zero) * box_scale;
                anchor_ptr[i] = anchor[i] * anchor_scale;
            }
            // FILE* pf = fopen("score","w");
            for(int i = 0; i < num_boxes * (num_classes); i++)
            {
                score_ptr[i] = score[i] * score_scale;
                //    if(i%91 ==0) fprintf(pf,"\n[%d]:",i/91);
                //    fprintf(pf,"%f,",score_ptr[i]);
            }
            // fclose(pf);
        }

        Box selected_box;
        for(int j = 0; j < num_boxes; j++)
        {
            for(int i = 1; i < num_classes; i++)
            {
                float score = score_ptr[j * (num_classes) + i];

                if(score < 0.6)
                    continue;

                selected_box.score = score;
                selected_box.class_idx = i;
                selected_box.box_idx = j;

                if(decode_single_box(selected_box, box_ptr, anchor_ptr, scales) < 0)
                    continue;

                auto& cls_vector = all_class_bbox_rects.at(i);
                cls_vector.emplace_back(selected_box);
            }
        }
        if(sizeof(type) == 1)
        {
            std::free(anchor_ptr);
            std::free(score_ptr);
            std::free(box_ptr);
        }
    }

    bool Run(Node* node)
    {
        Tensor* input_box_encodings = node->GetInputTensor(0);
        Tensor* input_class_predictions = node->GetInputTensor(1);
        Tensor* input_anchors = node->GetInputTensor(2);

        Tensor* output_detection_boxes = node->GetOutputTensor(0);
        Tensor* output_detection_classes = node->GetOutputTensor(1);
        Tensor* output_detection_scores = node->GetOutputTensor(2);
        Tensor* output_num_detections = node->GetOutputTensor(3);

        DetectionPostProcess* detect_op = dynamic_cast<DetectionPostProcess*>(node->GetOp());
        DetectionPostProcessParam* param = detect_op->GetParam();
        int elem_size = DataType::GetTypeSize(input_box_encodings->GetDataType());

        uint8_t* box_ptr = ( uint8_t* )get_tensor_mem(input_box_encodings);
        uint8_t* score_ptr = ( uint8_t* )get_tensor_mem(input_class_predictions);
        uint8_t* anchor_ptr = ( uint8_t* )get_tensor_mem(input_anchors);

        float* detection_boxes = ( float* )get_tensor_mem(output_detection_boxes);
        float* detection_classes = ( float* )get_tensor_mem(output_detection_classes);
        float* detection_scores = ( float* )get_tensor_mem(output_detection_scores);
        float* num_detections = ( float* )get_tensor_mem(output_num_detections);

        const std::vector<int>& dims = input_box_encodings->GetShape().GetDim();
        const int num_boxes = dims[1];
        const int num_classes = param->num_classes + 1;
        const int max_detections = param->max_detections;

        // printf("num_box: %d ,num_classes: %d ,max: %d \n",num_boxes,num_classes,max_detections);

        std::vector<std::vector<Box>> all_class_bbox_rects;
        std::vector<float>& scales = param->scales;

        all_class_bbox_rects.resize(num_classes);

        if(elem_size == 4)
        {
            get_all_boxes_rect<float>(all_class_bbox_rects, box_ptr, score_ptr, anchor_ptr, 1, 1, 1, num_boxes,
                                      num_classes, scales);
        }
        else if(elem_size == 1)
        {
            auto box_quant = input_box_encodings->GetQuantParam();
            float box_scale = (*box_quant)[0].scale;
            box_zero = (*box_quant)[0].zero_point;
            auto score_quant = input_class_predictions->GetQuantParam();
            float score_scale = (*score_quant)[0].scale;
            score_zero = (*score_quant)[0].zero_point;
            auto anchor_quant = input_anchors->GetQuantParam();
            float anchor_scale = (*anchor_quant)[0].scale;
            anchor_zero = (*anchor_quant)[0].zero_point;
            get_all_boxes_rect<uint8_t>(all_class_bbox_rects, box_ptr, score_ptr, anchor_ptr, box_scale, score_scale,
                                        anchor_scale, num_boxes, num_classes, scales);
        }

        std::vector<Box> all_boxes;

        for(int i = 1; i < num_classes; i++)
        {
            std::vector<Box>& class_box = all_class_bbox_rects.at(i);

            if(class_box.empty())
                continue;

            // sort
            std::sort(class_box.begin(), class_box.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

            if(( int )class_box.size() > max_detections * 2)
                class_box.resize(max_detections * 2);

            std::vector<int> picked;
            nms_sorted_bboxes(class_box, picked, param->nms_iou_threshold);

            // save the survivors
            for(int j = 0; j < ( int )picked.size(); j++)
            {
                int z = picked[j];
                all_boxes.emplace_back(class_box[z]);
            }
        }

        std::sort(all_boxes.begin(), all_boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        if(max_detections < ( int )all_boxes.size())
            all_boxes.resize(max_detections);

        // generate output tensors

        num_detections[0] = all_boxes.size();

        for(unsigned int i = 0; i < all_boxes.size(); i++)
        {
            Box& box = all_boxes[i];

            detection_classes[i] = box.class_idx;
            detection_scores[i] = box.score;

            detection_boxes[4 * i] = box.x0;
            detection_boxes[4 * i + 1] = box.y0;
            detection_boxes[4 * i + 2] = box.x1;
            detection_boxes[4 * i + 3] = box.y1;
        }

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if((data_type != TENGINE_DT_FP32&&data_type != TENGINE_DT_UINT8) ||
        exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    DetectionPostProcessOps* ops = new DetectionPostProcessOps();

    return ops;
}

}    // namespace DetectionPostProcessImpl

using namespace DetectionPostProcessImpl;

void RegisterDetectionPostProcessNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "DetectionPostProcess", DetectionPostProcessImpl::SelectFunc, 1000);
}

}    // namespace TEngine
