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

#include "acl_executor.hpp"

extern "C"
{
#include "operator/op.h"
#include "utility/sys_port.h"

#include "convolution_param.h"
}

bool CLGraph::AddConvolutionLayer(struct node* node)
{
    TLOG_INFO("Tengine ACl: Support OP(%d) OP_CONV.\n", node->index);
    struct graph* graph = node->graph;
    void* acl_data = nullptr;
    void* data = nullptr;
    void* scratch_mem = NULL;
    ActivationLayerInfo act_info;
    struct conv_param* param = ( struct conv_param* )node->op.param_mem;

    if (param->activation == 0)
        act_info = ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU);
    if (param->activation == 6)
        act_info = ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU);

    int pad_x = param->pad_w0;
    int pad_y = param->pad_h0;
    int pad_x_1 = param->pad_w1;
    int pad_y_1 = param->pad_h1;
    int stride_x = param->stride_w;
    int stride_y = param->stride_h;
    int dilation_x = param->dilation_w;
    int dilation_y = param->dilation_h;
    int group = param->group;
    int outchan = param->output_channel;

    /* input */
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    char* name = input_tensor->name;

    CLTensor* itensor = nullptr;
    if (tensors_map_.count(name))
    {
        itensor = tensors_map_[name];
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
        TLOG_DEBUG("Can't find node [%s] tensor named :%s\n", node->name, name);
        return false;
    }

    /* bias */
    struct tensor* b_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
    CLTensor* btensor = nullptr;
    if (b_tensor && node->input_num > 2)
    {
        int* dim = b_tensor->dims;
        int channel = 1;
        for (int i = 0; i < b_tensor->dim_num; i++)
        {
            channel *= dim[i];
        }
        name = b_tensor->name;
        btensor = new CLTensor();
        btensor->allocator()->init(TensorInfo(TensorShape(channel, 1, 1, 1), 1, data_type_));
        tensors_map_[name] = btensor;
    }

    /* output */
    struct tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    int* dim_o = o_tensor->dims;
    name = o_tensor->name;
    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = o_tensor->layout;

    if (bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    else
    {
        // keep  the same datalayout
        assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
        otensor->allocator()->init(ClTensorInfo_o);
    }

    tensors_map_[name] = otensor;
    /* weight */
    struct tensor* w_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    int* dim_w = w_tensor->dims;
    int TengineWightDataLayOut = w_tensor->layout;
    name = w_tensor->name;

    CLTensor* wtensor = new CLTensor();
    tensors_map_[name] = wtensor;

    /* configure */
    bool bPermuteFlag = false;
    if (group > 1 && group == outchan)
    {
        // 1. weight proc
        if (bForcedNHWCMode_ == true && TengineWightDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need permute
            void* pvBuf = w_tensor->data;
            int s32DataSize = w_tensor->elem_size * w_tensor->elem_num;
            int TengineDatatype = w_tensor->data_type;
            assert(TengineDatatype < MAX_TENGINE_DATA_TYPE_NUM);
            int s32TengineEleSize = gs32TengineDataElemetSize[TengineDatatype];
            assert(( int )(dim_w[0] * dim_w[1] * dim_w[2] * dim_w[3] * s32TengineEleSize) == s32DataSize);
            // if(s32DataSize > l32ScratchMemSize_)
            // {
            //     delete pcScratchMem_;
            //     pcScratchMem_ = new char[s32DataSize];
            //     // pcScratchMem_ = (char*)sys_realloc(pcScratchMem_, s32DataSize * sizeof(int));
            //     l32ScratchMemSize_ = s32DataSize;
            // }
            // assert(pcScratchMem_ != NULL);

            scratch_mem = sys_malloc(s32DataSize);
            assert(scratch_mem != NULL);

            _PermuteDataLayoutNCHWToNHWC(pvBuf, dim_w[1], dim_w[0], dim_w[2], dim_w[3], scratch_mem,
                                         s32TengineEleSize);
            TensorInfo w_info = TensorInfo(TensorShape(dim_w[0], dim_w[3], dim_w[2], dim_w[1]), 1, data_type_);
            w_info.set_data_layout(DataLayout::NHWC);
            wtensor->allocator()->init(w_info);
            bPermuteFlag = true;
        }

        else
        {
            // NCHW
            TensorInfo ClTensorInfo =
                    TensorInfo(TensorShape(dim_w[3], dim_w[2], dim_w[0], dim_w[1]), 1, data_type_);
            ClTensorInfo.set_data_layout(DataLayout::NCHW);
            wtensor->allocator()->init(ClTensorInfo);
        }

        if (3 == dim_w[2] && 3 == dim_w[3])
        {
            //            CLDepthwiseConvolutionLayer::CLDepthwiseConvolutionLayerInternal3x3* dwconv3x3 = new CLDepthwiseConvolutionLayerInternal3x3(); dwconv3x3->configure(itensor, wtensor, btensor, otensor,
            //                                 PadStrideInfo(stride_x, stride_y, pad_x, pad_y), 1, act_info);
            //            functions_map_.push_back(dwconv3x3);

            std::shared_ptr<CLDepthwiseConvolutionLayer> dwconv = std::make_shared<CLDepthwiseConvolutionLayer>();
            dwconv->configure(itensor, wtensor, btensor, otensor, PadStrideInfo(stride_x, stride_y, pad_x, pad_y), 1, act_info);
            functions_map_.push_back(dwconv);
        }
        else
        {
            if (act_info.enabled())
            {
                return false;
            }

            std::shared_ptr<CLDepthwiseConvolutionLayer> dwconv = std::make_shared<CLDepthwiseConvolutionLayer>();
            dwconv->configure(itensor, wtensor, btensor, otensor, PadStrideInfo(stride_x, stride_y, pad_x, pad_y));
            functions_map_.push_back(dwconv);
        }
    }
    else
    {
        // 1. weight proc
        if (bForcedNHWCMode_ == true && TengineWightDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need permute
            void* pvBuf = w_tensor->data;
            int s32DataSize = w_tensor->elem_size * w_tensor->elem_num;
            int TengineDatatype = w_tensor->data_type;
            assert(TengineDatatype < MAX_TENGINE_DATA_TYPE_NUM);
            int s32TengineEleSize = gs32TengineDataElemetSize[TengineDatatype];
            assert(( int )(dim_w[0] * dim_w[1] * dim_w[2] * dim_w[3] * s32TengineEleSize) == s32DataSize);

            // if(s32DataSize > l32ScratchMemSize_)
            // {
            //    delete pcScratchMem_;
            //     pcScratchMem_ = new char[s32DataSize];
            //     // pcScratchMem_ = (char*)sys_realloc(pcScratchMem_, s32DataSize * sizeof(int));
            //     l32ScratchMemSize_ = s32DataSize;
            // }
            // assert(pcScratchMem_ != NULL);

            scratch_mem = sys_malloc(s32DataSize);
            assert(scratch_mem != NULL);

            _PermuteDataLayoutNCHWToNHWC(pvBuf, dim_w[0], dim_w[1], dim_w[2], dim_w[3], scratch_mem,
                                         s32TengineEleSize);
            TensorInfo w_info = TensorInfo(TensorShape(dim_w[1], dim_w[3], dim_w[2], dim_w[0]), 1, data_type_);
            w_info.set_data_layout(DataLayout::NHWC);
            wtensor->allocator()->init(w_info);
            bPermuteFlag = true;
        }
        else
        {
            // NCHW
            TensorInfo ClTensorInfo =
                    TensorInfo(TensorShape(dim_w[3], dim_w[2], dim_w[1], dim_w[0]), 1, data_type_);
            ClTensorInfo.set_data_layout(DataLayout::NCHW);
            wtensor->allocator()->init(ClTensorInfo);
        }

        std::shared_ptr<CLConvolutionLayer> clconv = std::make_shared<CLConvolutionLayer>();
        if (bForcedNHWCMode_ == true)
        {
            clconv->configure(
                itensor, wtensor, btensor, otensor,
                PadStrideInfo(stride_x, stride_y, pad_x, pad_x_1, pad_y, pad_y_1, DimensionRoundingType::FLOOR),
                WeightsInfo(), Size2D(dilation_x, dilation_y), act_info);
        }
        else
        {
            clconv->configure(
                itensor, wtensor, btensor, otensor,
                PadStrideInfo(stride_x, stride_y, pad_x, pad_x_1, pad_y, pad_y_1, DimensionRoundingType::FLOOR),
                WeightsInfo(), Size2D(dilation_x, dilation_y), act_info, false, group);
        }

        functions_map_.push_back(clconv);
    }
    wtensor->allocator()->allocate();
    wtensor->map();
    assert(((bPermuteFlag == true) ^ (scratch_mem != NULL)) == 0);
    data = (bPermuteFlag == true) ? scratch_mem : w_tensor->data;

    acl_data = wtensor->buffer();
    int size = w_tensor->elem_size * w_tensor->elem_num;
    copy_buffer(acl_data, data, size, data_type_, DataType::F32);
    wtensor->unmap();
    if (btensor && node->input_num > 2)
    {
        btensor->allocator()->allocate();
        btensor->map();
        data = b_tensor->data;
        acl_data = btensor->buffer();
        int size = b_tensor->elem_size * b_tensor->elem_num;
        copy_buffer(acl_data, data, size, data_type_, DataType::F32);
        btensor->unmap();
    }

    if (!scratch_mem)
        sys_free(scratch_mem);

    return true;
}
