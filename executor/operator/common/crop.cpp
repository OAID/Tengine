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
 * Copyright (c) 2017, Open AI Lab
 * Author: bingzhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <complex>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "data_type.hpp"
#include "operator/crop.hpp"

namespace TEngine {

namespace CropImpl {

struct CropOps : public NodeOps
{
    bool Run(Node* node)
    {
        // Get data Info from input tensor
        Tensor* input_tensor = node->GetInputTensor(0);
        TShape& inShape = input_tensor->GetShape();
        float* input = ( float* )get_tensor_mem(input_tensor);
        std::vector<int> inDims = inShape.GetDim();
        int iDataH = inDims[2];
        int iDataW = inDims[3];
        int iDataC = inDims[1];
        // int iDataN = inDims[0];

        // Get data info from output tensor
        Tensor* output_tensor = node->GetOutputTensor(0);
        TShape& outShape = output_tensor->GetShape();
        float* output = ( float* )get_tensor_mem(output_tensor);
        std::vector<int> outDims = outShape.GetDim();
        int oDataH = outDims[2];
        int oDataW = outDims[3];
        int oDataC = outDims[1];
        int oDataN = outDims[0];

        // Get param infor from Crop param
        Crop* cp_op = dynamic_cast<Crop*>(node->GetOp());
        CropParam* param_ = cp_op->GetParam();
        if(param_->flag == 1){
            if(param_->num_args == 1)
            {
                int offsetH = (iDataH - param_->crop_h) / 2;
                int offsetW = (iDataW - param_->crop_w) / 2;
                if((param_->offset_h + oDataH <= iDataH) && (param_->offset_w + oDataW <= iDataW))
                {
                    for(int n = 0; n < oDataN; n++)
                    {
                        for(int c = 0; c < oDataC; c++)
                        {
                            for(int h = 0; h < oDataH; h++)
                            {
                                int i_h = h + offsetH;
                                for(int w = 0; w < oDataW; w++)
                                {
                                    int i_w = w + offsetW;
                                    output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                        input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                                }
                            }
                        }
                    }
                }
            }
            if(param_->num_args == 2)
            {
                if((param_->offset_h + oDataH <= iDataH) && (param_->offset_w + oDataW <= iDataW))
                {
                    for(int n = 0; n < oDataN; n++)
                    {
                        for(int c = 0; c < oDataC; c++)
                        {
                            for(int h = 0; h < oDataH; h++)
                            {
                                int i_h = h + param_->offset_h;
                                for(int w = 0; w < oDataW; w++)
                                {
                                    int i_w = w + param_->offset_w;
                                    output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                        input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                                }
                            }
                        }
                    }
                }
            }
        }
        if(param_->flag == 0){
            if(param_->axis == 1){
                for(int n = 0; n < oDataN; n++){
                    for(int c = 0; c < oDataC; c++){
                        int i_c = param_->offset_c + c;
                        for(int h = 0; h < oDataH; h++){
                            int i_h = param_->offset_h + h;
                            for(int w = 0; w < oDataW; w++){
                                int i_w = param_->offset_w + w;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                    input[n * iDataC * iDataH * iDataW + i_c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
            if(param_->axis == 2){
                for(int n = 0; n < oDataN; n++){
                    for(int c = 0; c < oDataC; c++){
                        for(int h = 0; h < oDataH; h++){
                            int i_h = param_->offset_h + h;
                            for(int w = 0; w < oDataW; w++){
                                int i_w = param_->offset_w + w;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] =
                                    input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
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

    CropOps* ops = new CropOps();

    return ops;
}
}    // namespace CropImpl

void RegisterCrop_NodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Crop", CropImpl::SelectFunc, 1000);
}
}    // namespace TEngine
