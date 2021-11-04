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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#include "../trt_executor.hpp"

EXPORT_BEGIN
#include "crop_param.h"
EXPORT_FINISH

#include <NvInferRuntime.h>


bool TensorRTEngine::AddCropNode(struct graph* ir_graph, struct node* node)
{
    struct tensor* crop_input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* crop_ref = get_ir_graph_tensor(ir_graph, node->input_tensors[1]);
    struct tensor* crop_output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    if (nullptr == crop_input || nullptr == crop_ref || nullptr == crop_output)
    {
        fprintf(stderr, "Tengine: Get input & output for Crop(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    if (!check_if_input_in_map(crop_input->index, this->tensor_swap_map))
    {
        fprintf(stderr, "Tengine: Query input for Crop(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    auto* param = (struct crop_param*)node->op.param_mem;

    param->axis = (param->axis < 0) ? 4 + param->axis : param->axis; // axis negative number correction

    // acceptable axis values: 2, 3, -1, -2
    // unacceptable axis values: 0, 1, -3, -4 and anything else
    // acceptable corrected axis values: 2, 3
    // unacceptable corrected axis values: 0, 1 and anything else
    // protect against "garbage" input arguments
    bool axis_abort = (param->axis != 2 && param->axis != 3);

    // get the offsets
    // the offsets are zero by default (in case no offset is specified)
    int offsetHeight = param->offset_h;
    int offsetWidth = param->offset_w;

    // now compute the prePadding and postPadding required to perform the crop
    // so that the first bottom is the same spatial size as the second bottom
    // prePadding is the padding to the left/bottom (assuming origin is lower-left).
    // postPadding is the padding to the right/top.
    // - ( inputDims.h() - refDims.h() - offsetHeight ) = -inputDims.h() + refDims.h() + offsetHeight
    // - ( inputDims.w() - refDims.w() - offsetWidth ) = -inputDims.w() + refDims.w() + offsetWidth
    int prePadHeight = -offsetHeight;
    int prePadWidth  = -offsetWidth;
    int postPadHeight = -crop_input->dims[2] + crop_ref->dims[2] + offsetHeight;
    int postPadWidth  = -crop_input->dims[3] + crop_ref->dims[3] + offsetWidth;

    nvinfer1::DimsHW prePadding(-param->offset_h, -param->offset_w);
    nvinfer1::DimsHW postPadding = nvinfer1::DimsHW{postPadHeight, postPadWidth};

    nvinfer1::ITensor* crop_input_tensor = tensor_real_map[tensor_swap_map[crop_input->index]];

    nvinfer1::IPaddingLayer* layer = this->network->addPadding(*crop_input_tensor, prePadding, postPadding);
    if (nullptr == layer)
    {
        fprintf(stderr, "Tengine: Add Crop(id: %d, name: %s) layer failed.\n", node->index, node->name);
        return false;
    }

    layer->setName(node->name);

    this->layer_map[node->index] = layer;

    nvinfer1::ITensor* crop_output_tensor = layer->getOutput(0);

    this->SetRange(crop_output, crop_output_tensor);

    this->tensor_real_map[node->output_tensors[0]] = crop_output_tensor;
    this->tensor_swap_map[node->output_tensors[0]] = node->output_tensors[0];

    return true;
}
