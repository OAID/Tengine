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
#include "crop_param.h"
}

bool CLGraph::AddCropLayer(struct node* node)
{
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name_in = input_tensor->name;
    int* dims_in = input_tensor->dims;
    struct tensor* crop_ref = get_ir_graph_tensor(graph, node->input_tensors[1]);

    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    std::string name_out = output_tensor->name;
    int* dims_out = output_tensor->dims;

    struct crop_param* param = ( struct crop_param* )node->op.param_mem;

    int offsetHeight = param->offset_h;
    int offsetWidth = param->offset_w;

    /* set acl input tensor */
    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name_in))
    {
        itensor = tensors_map_[name_in];
        if (bForcedNHWCMode_ == true)    //
        {
            TensorInfo* pClTensorInfo = itensor->info();
            if (pClTensorInfo->data_layout() == DataLayout::NCHW)
            {
                int* dim = input_tensor->dims;
                assert(input_tensor->dim_num == 4);

                // set acl dims : cwhn
                pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
                pClTensorInfo->set_data_layout(DataLayout::NHWC);
            }
            else
            {
                assert(pClTensorInfo->data_layout() == DataLayout::NHWC);
            }
        }
    }
    else
    {
        TLOG_DEBUG("Can't find node [%s] tensor named :%s\n", node->name, name_in.c_str());
        return false;
    }

    /* set acl output tensor */
    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = output_tensor->layout;

    TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dims_out[1], dims_out[3], dims_out[2], dims_out[0]), 1, data_type_);
    ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
    otensor->allocator()->init(ClTensorInfo_o);
    tensors_map_[name_out] = otensor;

    InterpolationPolicy upsampling_policy = InterpolationPolicy::NEAREST_NEIGHBOR;

    CLTensor* boxes = new CLTensor();
    TensorInfo ClTensorInfo_boxes = TensorInfo(TensorShape(4, 1), 1, data_type_);
    boxes->allocator()->init(ClTensorInfo_boxes);
    tensors_map_[std::to_string(node->index+0.1).c_str()] = boxes;
    float boxes_i[4] = {offsetHeight/(float)dims_in[2], offsetWidth/(float)dims_in[3],
                        (offsetHeight + dims_out[2])/(float)dims_in[2], (offsetWidth + dims_out[3])/(float)dims_in[3]};
    boxes->allocator()->allocate();
    boxes->map();
    void* boxes_tensor_data = boxes->buffer();
    copy_buffer(boxes_tensor_data, boxes_i, 4 * sizeof(float), DataType::F32, DataType::F32);
    boxes->unmap();

    CLTensor* box_ind = new CLTensor();
    TensorInfo ClTensorInfo_boxesidx = TensorInfo(TensorShape(1), 1, DataType::F32);
    box_ind->allocator()->init(ClTensorInfo_boxesidx);
    tensors_map_[std::to_string(node->index+0.2).c_str()] = box_ind;
    int boxes_idx[1] = {0};
    box_ind->allocator()->allocate();
    box_ind->map();
    void* boxes_idx_tensor_data = box_ind->buffer();
    copy_buffer(boxes_idx_tensor_data, boxes_idx, sizeof(int), DataType::F32, DataType::F32);
    box_ind->unmap();

    Coordinates2D crop_size;
    crop_size.x = dims_out[3];
    crop_size.y = dims_out[2];

    tensors_map_[name_out] = otensor;

    /* add Corp layer into acl graph */
    std::shared_ptr<CLCropResize> interp = std::make_shared<CLCropResize>();

    if (data_type_ == DataType::F16)
    {
        /* initial cast temp tensor */
        int* dim_in = input_tensor->dims;
        CLTensor* cast_in_tensor = new CLTensor();
        TensorInfo ClTensorInfo_cast_in = TensorInfo(TensorShape(dim_in[1], dim_in[3], dim_in[2], dim_in[0]), 1, DataType::F32);
        ClTensorInfo_cast_in.set_data_layout(DataLayout::NHWC);
        cast_in_tensor->allocator()->init(ClTensorInfo_cast_in);
        tensors_map_[std::to_string(node->index+0.1).c_str()] = cast_in_tensor;

        int* dim_out = output_tensor->dims;
        CLTensor* cast_out_tensor = new CLTensor();
        TensorInfo ClTensorInfo_cast_out = TensorInfo(TensorShape(dim_out[1], dim_out[3], dim_out[2], dim_out[0]), 1, DataType::F32);
        ClTensorInfo_cast_out.set_data_layout(DataLayout::NHWC);
        cast_out_tensor->allocator()->init(ClTensorInfo_cast_out);
        tensors_map_[std::to_string(node->index+0.2).c_str()] = cast_out_tensor;

        /* fp16 to fp32 */
        std::shared_ptr<CLCast> cast_in = std::make_shared<CLCast>();
        std::shared_ptr<CLCast> cast_out = std::make_shared<CLCast>();
        cast_in->configure(itensor, cast_in_tensor, ConvertPolicy::WRAP);
        interp->configure(cast_in_tensor, boxes, box_ind, cast_out_tensor, crop_size, upsampling_policy);
        cast_out->configure(cast_out_tensor, otensor, ConvertPolicy::WRAP);

        functions_map_.push_back(cast_in);
        functions_map_.push_back(interp);
        functions_map_.push_back(cast_out);
    }
    else
    {
        interp->configure(itensor, boxes, box_ind, otensor, crop_size, upsampling_policy);
        functions_map_.push_back(interp);
    }

    return true;
}

