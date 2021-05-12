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
 * Author: hhchen@openailab.com
 */


#include "acl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "interp_param.h"
}

bool CLGraph::AddInterpLayer(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name_in = input_tensor->name;
    int* dims_in = input_tensor->dims;

    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    std::string name_out = output_tensor->name;
    int* dims_out = output_tensor->dims;

    struct interp_param* param = ( struct interp_param* )node->op.param_mem;

//    fprintf(stderr,"param->resize_type %d\n",param->resize_type);
    InterpolationPolicy upsampling_policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    if (param->resize_type == 2)
        upsampling_policy = InterpolationPolicy::BILINEAR;

    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name_in))
    {
        itensor = tensors_map_[name_in];
    }
    else
    {
        return false;
    }

    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = output_tensor->layout;

    TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dims_out[1], dims_out[3], dims_out[2], dims_out[0]), 1, data_type_);
    ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
    otensor->allocator()->init(ClTensorInfo_o);

    tensors_map_[name_out] = otensor;

    std::shared_ptr<CLScale> interp = std::make_shared<CLScale>();
    interp->configure(itensor, otensor, upsampling_policy, BorderMode::UNDEFINED);
    functions_map_.push_back(interp);

    return true;
}

//
//bool CLGraph::AddInterpLayer(struct node* node)
//{
//    struct graph* graph = node->graph;
//    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
//    std::string name_in = input_tensor->name;
//    int* dims_in = input_tensor->dims;
//
//    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
//    std::string name_out = output_tensor->name;
//    int* dims_out = output_tensor->dims;
//
//    struct interp_param* param = ( struct interp_param* )node->op.param_mem;
//
//    fprintf(stderr,"param->resize_type %d\n",param->resize_type);
//    InterpolationPolicy upsampling_policy = InterpolationPolicy::NEAREST_NEIGHBOR;
//    if (param->resize_type == 2)
//        upsampling_policy = InterpolationPolicy::BILINEAR;
//
//    float height_scale = param->height_scale;
//    float width_scale = param->width_scale;
//    fprintf(stderr,"Log: h w: %f, %f\n", height_scale, width_scale);
//
//    CLTensor* boxes = new CLTensor();
//    TensorInfo ClTensorInfo_boxes = TensorInfo(TensorShape(4, 1), 1, data_type_);
//    boxes->allocator()->init(ClTensorInfo_boxes);
//    tensors_map_[std::to_string(node->index+0.1).c_str()] = boxes;
//    float boxes_i[4] = {0, 0, 1, 1};
//    boxes->allocator()->allocate();
//    boxes->map();
//    void* boxes_tensor_data = boxes->buffer();
//    copy_buffer(boxes_tensor_data, boxes_i, 4 * sizeof(float), data_type_, DataType::F32);
//    boxes->unmap();
//
//
//    CLTensor* box_ind = new CLTensor();
//    TensorInfo ClTensorInfo_boxesidx = TensorInfo(TensorShape(1), 1, data_type_);
//    box_ind->allocator()->init(ClTensorInfo_boxesidx);
//    tensors_map_[std::to_string(node->index+0.2).c_str()] = box_ind;
//    int boxes_idx[1] = {0};
//    box_ind->allocator()->allocate();
//    box_ind->map();
//    void* boxes_idx_tensor_data = box_ind->buffer();
//    copy_buffer(boxes_idx_tensor_data, boxes_idx, sizeof(int), data_type_, DataType::F32);
//    box_ind->unmap();
//
//
//    CLTensor* itensor = nullptr;
//    if (tensors_map_.count(name_in))
//    {
//        itensor = tensors_map_[name_in];
//    }
//    else
//    {
//        return false;
//    }
//
//    CLTensor* otensor = new CLTensor();
//    int TengineDataLayOut = output_tensor->layout;
//
//    TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dims_out[1], dims_out[3], dims_out[2], dims_out[0]), 1, data_type_);
//    ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
//    otensor->allocator()->init(ClTensorInfo_o);
//
//    Coordinates2D crop_size;
//    crop_size.x = dims_out[3];
//    crop_size.y = dims_out[2];
//
//    tensors_map_[name_out] = otensor;
//    CLCropResize* interp = new CLCropResize();
//    interp->configure(itensor, boxes, box_ind, otensor, crop_size, upsampling_policy);
//    functions_map_.push_back(interp);
//
//    return true;
//}
