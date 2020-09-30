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
 * Copyright (c) 2020, Open AI Lab
 * Author: xlchen@openailab.com
 */
#ifndef __ACL_GRAPH_HPP__
#define __ACL_GRAPH_HPP__

#include <array>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>
#include <arm_neon.h>

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

extern "C" {
    //#include "compiler_fp16.h"
    #include "tengine_errno.h"
    #include "tengine_log.h"
    #include "convolution_param.h"
    #include "pooling_param.h"
    #include "batchnorm_param.h"
    #include "eltwise_param.h"
    #include "relu_param.h"
}

using namespace arm_compute;

#define USE_CPU_CONVERT
//#define ACL_EXTENSTION
#ifdef __ANDROID__
#define dynamic_cast static_cast
#endif

static inline void copy_fp32_to_fp16(__fp16* f16, const float* f32, const int f32_size)
{
    for(unsigned int i = 0; i < f32_size / sizeof(float); i++)
        f16[i] = f32[i];
}

static inline void copy_fp16_to_fp32(float* f32, const __fp16* f16, const int f16_size)
{
    for(unsigned int i = 0; i < f16_size / sizeof(__fp16); i++)
        f32[i] = f16[i];
}

inline void copy_buffer(void* dest, const void* src, const int src_len, DataType dest_type, DataType src_type)
{
    if(dest_type == src_type)
        memcpy(dest, src, src_len);
    else if(dest_type == DataType::F16 && src_type == DataType::F32)
        copy_fp32_to_fp16(( __fp16* )dest, ( const float* )src, src_len);
    else if(dest_type == DataType::F32 && src_type == DataType::F16)
        copy_fp16_to_fp32(( float* )dest, ( const __fp16* )src, src_len);
    else
        printf("copy_buffer may failed!!!");
}

#define MAX_TENGINE_DATA_TYPE_NUM 6
static const int gs32TengineDataElemetSize[MAX_TENGINE_DATA_TYPE_NUM] = {4, 2, 1, 1, 4, 2};

template <typename T>
inline void _PermuteDatalayoutNCHWToNHWCInter(T* pvData, int n, int c, int h, int w, T* pvOutputData)
{
    T* pDataInputBuf = pvData;
    T* pDataOutputBuf = pvOutputData;
    int s32Cnt = 0;
    for(int z = 0; z < n; z++)
    {
        for(int i = 0; i < h; i++)
        {
            const T* pRowStartAddr = pDataInputBuf + w * i + z * w * h * c;
            for(int j = 0; j < w; j++)
            {
                for(int k = 0; k < c; k++)
                {
                    const T* pCkData = pRowStartAddr + k * (w * h) + j;
                    pDataOutputBuf[s32Cnt] = *pCkData;
                    s32Cnt++;
                }
            }
        }
    }
}

inline void _PermuteDatalayoutNCHWToNHWC(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize)
{
    assert(pvData != NULL);
    assert(pvOutputData != NULL);
    assert(DataEleSize == 1 || DataEleSize == 2 || DataEleSize == 4);
    if(DataEleSize == 4)
    {
        _PermuteDatalayoutNCHWToNHWCInter(( int* )pvData, n, c, h, w, ( int* )pvOutputData);
    }
    else if(DataEleSize == 2)
    {
        _PermuteDatalayoutNCHWToNHWCInter(( short* )pvData, n, c, h, w, ( short* )pvOutputData);
    }
    else
    {
        _PermuteDatalayoutNCHWToNHWCInter(( char* )pvData, n, c, h, w, ( char* )pvOutputData);
    }
}

template <typename T>
inline void _PermuteDatalayoutNHWCToNCHWInter(T* pvData, int n, int c, int h, int w, T* pvOutputData)
{
    T* pDataInputBuf = pvData;
    T* pDataOutputBuf = pvOutputData;
    int s32Cnt = 0;
    for(int z = 0; z < n; z++)
    {
        for(int i = 0; i < h; i++)
        {
            T* pRowStartAddr = pDataOutputBuf + w * i + z * w * h * c;
            for(int j = 0; j < w; j++)
            {
                for(int k = 0; k < c; k++)
                {
                    T* pCkData = pRowStartAddr + k * (w * h) + j;
                    *pCkData = pDataInputBuf[s32Cnt];
                    s32Cnt++;
                }
            }
        }
    }
}

inline void _PermuteDatalayoutNHWCToNCHW(void* pvData, int n, int c, int h, int w, void* pvOutputData, int DataEleSize)
{
    assert(pvData != NULL);
    assert(pvOutputData != NULL);
    assert(DataEleSize == 1 || DataEleSize == 2 || DataEleSize == 4);

    if(DataEleSize == 4)
    {
        _PermuteDatalayoutNHWCToNCHWInter(( int* )pvData, n, c, h, w, ( int* )pvOutputData);
    }
    else if(DataEleSize == 2)
    {
        _PermuteDatalayoutNHWCToNCHWInter(( short* )pvData, n, c, h, w, ( short* )pvOutputData);
    }
    else
    {
        _PermuteDatalayoutNHWCToNCHWInter(( char* )pvData, n, c, h, w, ( char* )pvOutputData);
    }
}

class MYSoftmaxLayer : public IFunction
{
public:
    CLSoftmaxLayer _layer;
    CLTensor* input_org;
    CLTensor _input;
    DataType data_type_;

    void configure(CLTensor* input, CLTensor* output, DataType type)
    {
        input_org = input;
        TensorInfo* info = input->info();
        int size = info->dimension(0) * info->dimension(1) * info->dimension(2);
        TensorShape shape = TensorShape(size);
        _input.allocator()->init(TensorInfo(shape, 1, type));
        _layer.configure(&_input, output);
        _input.allocator()->allocate();
        data_type_ = type;
    }
    void run()
    {
        TensorInfo* info = input_org->info();
        int size = info->dimension(0) * info->dimension(1) * info->dimension(2);
        input_org->map();
        _input.map();
        if(data_type_ == DataType::F32)
        {
            float* src = reinterpret_cast<float*>(input_org->buffer());
            float* dst = reinterpret_cast<float*>(_input.buffer());
            int w_align = info->dimension(1) + info->padding().right;
            for(int i = 0; i < size; i++)
            {
                dst[i] = src[i * w_align];
            }
        }
        else
        {
            __fp16* src = (__fp16*)input_org->buffer();
            __fp16* dst = (__fp16*)_input.buffer();
            int w_align = info->dimension(1) + info->padding().right;
            for(int i = 0; i < size; i++)
            {
                dst[i] = src[i * w_align];
            }
        }
        input_org->unmap();
        _input.unmap();
        _layer.run();
    }
};

class CLGraph
{
public:
    std::string name_;
    std::vector<IFunction*> functions_map_;
    std::unordered_map<std::string, CLTensor*> tensors_map_;
    DataType data_type_;

    int nCnt_ = 0;
    bool bForcedNHWCMode_;
    char* pcScratchMem_;
    int l32ScratchMemSize_;

    CLGraph(std::string name, DataType type)
    {
        name_ = name;
        data_type_ = type;
        bForcedNHWCMode_ = false;
        pcScratchMem_ = new char[8];
        l32ScratchMemSize_ = 0;
    };

    ~CLGraph()
    {
        delete pcScratchMem_;
    }

    void Run(void)
    {
        int size = functions_map_.size();
        for(int i = 0; i < size; i++)
        {
            functions_map_[i]->run();
        }
    }

    bool AddInputLayer(struct ir_node* node)
    {
        /* output */
        struct ir_graph* graph = node->graph;
        struct ir_tensor* tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        char* name = tensor->name;
        int* dim_w = tensor->dims;
        CLTensor* otensor = new CLTensor();
        TensorInfo ClTensorInfo = TensorInfo(TensorShape(dim_w[2], dim_w[3], dim_w[1], dim_w[0]), 1, data_type_);
        DataLayout aclDataLayout;
        aclDataLayout =
            (tensor->layout == 0) ? DataLayout::NCHW : DataLayout::NHWC;
        ClTensorInfo.set_data_layout(aclDataLayout);
        otensor->allocator()->init(ClTensorInfo);
        tensors_map_[name] = otensor;

        return true;
    }

    bool AddBNLayer(struct ir_node* node, struct ir_node* node_scale)
    {
        struct ir_graph* graph = node->graph;
        struct ir_graph* scale_graph = node_scale->graph;
        struct batchnorm_param* param = (struct batchnorm_param*)node->op.param_mem;
        float eps = param->eps;

        /* input */
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        char* name = input_tensor->name;
        int channel = input_tensor->dims[1];
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor->dims;
                    assert(input_tensor->dim_num == 4);

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
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        
        /* gamma */
        struct ir_tensor* gamma_tensor = get_ir_graph_tensor(scale_graph, node_scale->input_tensors[1]);
        CLTensor* gtensor = nullptr;
        if(gamma_tensor)
        {
            name = gamma_tensor->name;
            gtensor = new CLTensor();
            gtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = gtensor;
        }
        /* beta */
        struct ir_tensor* beta_tensor = get_ir_graph_tensor(scale_graph, node_scale->input_tensors[2]);
        CLTensor* btensor = nullptr;
        if(beta_tensor)
        {
            name = beta_tensor->name;
            btensor = new CLTensor();
            btensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = btensor;
        }

        /* means */
        struct ir_tensor* means_tensor = get_ir_graph_tensor(graph, node_scale->input_tensors[3]);
        name = means_tensor->name;
        CLTensor* mtensor = new CLTensor();
        mtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
        tensors_map_[name] = mtensor;

        /* var */
        struct ir_tensor* var_tensor = get_ir_graph_tensor(graph, node_scale->input_tensors[4]);
        CLTensor* vtensor = nullptr;
        if(var_tensor)
        {
            name = var_tensor->name;
            vtensor = new CLTensor();
            vtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = vtensor;
        }
        /* output */
        struct ir_tensor* out_tensor = get_ir_graph_tensor(graph, node_scale->output_tensors[0]);
        int* dim_o = out_tensor->dims;
        name = out_tensor->name;
        CLTensor* otensor = new CLTensor();

        int TengineDataLayOut = out_tensor->layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
            otensor->allocator()->init(ClTensorInfo_o);
        }
        else
        {
            // keep  the same datalayout
            assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
            // dim_o[3], dim_o[2], dim_o[1], dim_o[0]
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
            otensor->allocator()->init(ClTensorInfo_o);
        }

        tensors_map_[name] = otensor;

        CLBatchNormalizationLayer* bn = new CLBatchNormalizationLayer();
        bn->configure(itensor, otensor, mtensor, vtensor, btensor, gtensor, eps);

        functions_map_.push_back(bn);

        mtensor->allocator()->allocate();
        vtensor->allocator()->allocate();
        mtensor->map();
        vtensor->map();
        void* means_data = mtensor->buffer();
        void* vars_data = vtensor->buffer();
        void* means = means_tensor->data;
        void* vars = var_tensor->data;

        copy_buffer(means_data, means, channel * 4, data_type_, DataType::F32);
        copy_buffer(vars_data, vars, channel * 4, data_type_, DataType::F32);

        mtensor->unmap();
        vtensor->unmap();

        if(btensor)
        {
            btensor->allocator()->allocate();
            btensor->map();
            void* beta_data = btensor->buffer();
            void* beta = beta_tensor->data;
            copy_buffer(beta_data, beta, channel * 4, data_type_, DataType::F32);
            btensor->unmap();
        }
        if(gtensor)
        {
            gtensor->allocator()->allocate();
            gtensor->map();
            void* gamma_data = gtensor->buffer();
            void* gamma = gamma_tensor->data;
            copy_buffer(gamma_data, gamma, channel * 4, data_type_, DataType::F32);
            gtensor->unmap();
        }

        return true;
    }

    bool AddConcatLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        std::vector<ICLTensor*> inputs_vector;
        for(unsigned int i = 0; i < node->input_num; i++)
        {
            struct ir_tensor* tensor = get_ir_graph_tensor(graph, node->input_tensors[i]);
            char* name = tensor->name;
            CLTensor* itensor = nullptr;
            if(tensors_map_.count(name))
            {
                itensor = tensors_map_[name];
                if(bForcedNHWCMode_ == true)    //
                {
                    TensorInfo* pClTensorInfo = itensor->info();
                    if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                    {
                        int* dim = tensor->dims;
                        assert(tensor->dim_num == 4);

                        pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
                        pClTensorInfo->set_data_layout(DataLayout::NHWC);
                    }
                    else
                    {
                    }
                }
            }
            else
            {
                //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
                return false;
            }
            inputs_vector.push_back(itensor);
        }

        /*output */
        struct ir_tensor* out = get_ir_graph_tensor(graph, node->output_tensors[0]);
        int* dim_o = out->dims;
        char* name = out->name;
        CLTensor* otensor = new CLTensor();
        int TengineDataLayOut = out->layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
            otensor->allocator()->init(ClTensorInfo_o);
        }
        else
        {
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
            otensor->allocator()->init(ClTensorInfo_o);
        }
        tensors_map_[name] = otensor;

        CLConcatenateLayer* concat = new CLConcatenateLayer();
        // concat->configure(inputs_vector, otensor, DataLayoutDimension::CHANNEL);
        concat->configure(inputs_vector, otensor, 0);
        functions_map_.push_back(concat);
        return true;
    }

    bool AddConvolutionLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        void* acl_data = nullptr;
        void* data = nullptr;
        void* scratch_mem = NULL;
        ActivationLayerInfo act_info;
        struct conv_param* param = (struct conv_param*)node->op.param_mem;

        if(param->activation==0)
            act_info = ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU);
        if(param->activation==6)
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
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        char* name = input_tensor->name;

        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor->dims;
                    assert(input_tensor->dim_num == 4);

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
        struct ir_tensor* b_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
        CLTensor* btensor = nullptr;
        if(b_tensor && node->input_num > 2)
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
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        int* dim_o = o_tensor->dims;
        name = o_tensor->name;
        CLTensor* otensor = new CLTensor();
        int TengineDataLayOut = o_tensor->layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
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
        struct ir_tensor* w_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
        int* dim_w = w_tensor->dims;
        int TengineWightDataLayOut = w_tensor->layout;
        name = w_tensor->name;

        CLTensor* wtensor = new CLTensor();
        tensors_map_[name] = wtensor;
        /* configure */
        bool bPermuteFlag = false;
        if(group > 1 && group == outchan)
        {
            // 1. weight proc
            if(bForcedNHWCMode_ == true && TengineWightDataLayOut == TENGINE_LAYOUT_NCHW)
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

                _PermuteDatalayoutNCHWToNHWC(pvBuf, dim_w[1], dim_w[0], dim_w[2], dim_w[3], scratch_mem,
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

            if(3 == dim_w[2] && 3 == dim_w[3])
            {
                CLDepthwiseConvolutionLayer3x3* dwconv3x3 = new CLDepthwiseConvolutionLayer3x3();
                dwconv3x3->configure(itensor, wtensor, btensor, otensor,
                                     PadStrideInfo(stride_x, stride_y, pad_x, pad_y), 1, act_info);
                functions_map_.push_back(dwconv3x3);
            }
            else
            {
                if(act_info.enabled())
                    return false;
                CLDepthwiseConvolutionLayer* dwconv = new CLDepthwiseConvolutionLayer();
                dwconv->configure(itensor, wtensor, btensor, otensor, PadStrideInfo(stride_x, stride_y, pad_x, pad_y));
                functions_map_.push_back(dwconv);
            }
        }
        else
        {
            // 1. weight proc
            if(bForcedNHWCMode_ == true && TengineWightDataLayOut == TENGINE_LAYOUT_NCHW)
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

                _PermuteDatalayoutNCHWToNHWC(pvBuf, dim_w[0], dim_w[1], dim_w[2], dim_w[3], scratch_mem,
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
            CLConvolutionLayer* clconv = new CLConvolutionLayer();
            if(bForcedNHWCMode_ == true)
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
        if(btensor && node->input_num > 2)
        {
            btensor->allocator()->allocate();
            btensor->map();
            data = b_tensor->data;
            acl_data = btensor->buffer();
            int size = b_tensor->elem_size * b_tensor->elem_num;
            copy_buffer(acl_data, data, size, data_type_, DataType::F32);
            btensor->unmap();
        }

        if(!scratch_mem)
            sys_free(scratch_mem);

        return true;
    }

    bool AddDropoutLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor->dims;
                    assert(input_tensor->dim_num == 4);

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
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        /*output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        name = o_tensor->name;
        tensors_map_[name] = itensor;

        return true;
    }

    bool AddEltwiseLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor0 = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor0->name;
        CLTensor* itensor0 = nullptr;
        if(tensors_map_.count(name))
        {
            itensor0 = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor0->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor0->dims;
                    assert(input_tensor0->dim_num == 4);

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
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }
        struct ir_tensor* input_tensor1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
        name = input_tensor1->name;
        CLTensor* itensor1 = nullptr;
        if(tensors_map_.count(name))
        {
            itensor1 = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor1->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor1->dims;
                    assert(input_tensor1->dim_num == 4);

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
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }
        /*output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        name = o_tensor->name;
        int* dim_o = o_tensor->dims;
        CLTensor* otensor = new CLTensor();
        int TengineDataLayOut = o_tensor->layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
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

        struct eltwise_param* param = (struct eltwise_param*)node->op.param_mem;
        if(ELT_SUM == param->type)
        {
            CLArithmeticAddition* add = new CLArithmeticAddition();
            add->configure(itensor0, itensor1, otensor, ConvertPolicy::WRAP);
            functions_map_.push_back(add);
        }
        else
        {
            //TLOG_ERR("eltwise only support ADD!~~\n");
            return false;
        }

        return true;
    }

    bool AddFCLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        /* Input */
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor->dims;
                    assert(input_tensor->dim_num == 4);

                    pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
                    pClTensorInfo->set_data_layout(DataLayout::NHWC);
                }
                else
                {
                    assert(pClTensorInfo->data_layout() == DataLayout::NHWC);
                }
            }
        }
        if(!itensor)
        {
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }
        /* weight */
        struct ir_tensor* w_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
        name = w_tensor->name;
        int M = w_tensor->dims[0];
        int K = w_tensor->dims[1];
        CLTensor* wtensor = new CLTensor();
        wtensor->allocator()->init(TensorInfo(TensorShape(K, M), 1, data_type_));
        tensors_map_[name] = wtensor;
        /* bias */
        struct ir_tensor* b_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
        CLTensor* btensor = nullptr;

        if(b_tensor)
        {
            name = b_tensor->name;
            btensor = new CLTensor();
            btensor->allocator()->init(TensorInfo(TensorShape(M), 1, data_type_));
            tensors_map_[name] = btensor;
        }

        /*output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        name = o_tensor->name;
        int* dim_w = o_tensor->dims;
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim_w[1]), 1, data_type_));
        tensors_map_[name] = otensor;
        

        /* FC Layer */
        bool transpose_w = (dim_w[1] == M) ? true : false;
        CLFullyConnectedLayer* fc = new CLFullyConnectedLayer();
        FullyConnectedLayerInfo fc_info;
        fc_info.set_transpose_weights(transpose_w);
        fc_info.set_weights_trained_layout(DataLayout::NCHW);    // lay out
        fc->configure(itensor, wtensor, btensor, otensor, fc_info);
        functions_map_.push_back(fc);
        wtensor->allocator()->allocate();
        wtensor->map();
        void* data = w_tensor->data;
        void* acl_data = wtensor->buffer();
        int size = w_tensor->elem_size * w_tensor->elem_num;
        copy_buffer(acl_data, data, size, data_type_, DataType::F32);
        wtensor->unmap();
        if(btensor)
        {
            btensor->allocator()->allocate();
            btensor->map();
            data = b_tensor->data;
            acl_data = btensor->buffer();
            int size = b_tensor->elem_size * b_tensor->elem_num;
            copy_buffer(acl_data, data, size, data_type_, DataType::F32);
            btensor->unmap();
        }
        return true;
    }

    bool AddPoolingLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct pool_param* param = (struct pool_param*)node->op.param_mem;
        int pad_x = param->pad_w0;
        int pad_y = param->pad_h0;
        int stride_x = param->stride_w;
        int stride_y = param->stride_h;
        int kernel_w = param->kernel_w;
        int kernel_h = param->kernel_h;
        int type = param->pool_method;
        int global = param->global;

        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        int channel = input_tensor->dims[1];
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if(bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if(pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = input_tensor->dims;
                    assert(input_tensor->dim_num == 4);

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
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        /* output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

        int TengineDataLayOut = o_tensor->layout;
        TensorInfo* info = itensor->info();
        int out_h = std::ceil(( float )(info->dimension(1) - kernel_h + 2 * pad_y) / stride_y) + 1;
        int out_w = std::ceil(( float )(info->dimension(0) - kernel_w + 2 * pad_x) / stride_x) + 1;
        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            out_h = std::ceil(( float )(info->dimension(2) - kernel_h + 2 * pad_y) / stride_y) + 1;
            out_w = std::ceil(( float )(info->dimension(1) - kernel_w + 2 * pad_x) / stride_x) + 1;
        }
        name = o_tensor->name;
        int* dim_o = o_tensor->dims;
        CLTensor* otensor = new CLTensor();
        DataLayout data_layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(channel, out_w, out_h, 1), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
            otensor->allocator()->init(ClTensorInfo_o);
            data_layout = DataLayout::NHWC;
        }
        else
        {
            // keep  the same datalayout
            assert(TENGINE_LAYOUT_NCHW == TengineDataLayOut);
            // dim_o[3], dim_o[2], dim_o[1], dim_o[0]
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(out_w, out_h, channel, 1), 1, data_type_);
            ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
            otensor->allocator()->init(ClTensorInfo_o);
            data_layout = DataLayout::NCHW;
        }

        // otensor->allocator()->init(TensorInfo(TensorShape(out_h, out_w, channel, 1), 1, data_type_));
        tensors_map_[name] = otensor;
        CLPoolingLayer* pooling = new CLPoolingLayer();
        PoolingLayerInfo pooling_info;
        
        if(global)
            pooling_info = PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX, data_layout);
        else
            pooling_info =
                PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX, Size2D(kernel_w, kernel_h), data_layout, 
                                 PadStrideInfo(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::CEIL));

        pooling->configure(itensor, otensor, pooling_info);

        functions_map_.push_back(pooling);

        return true;
    }

    bool AddReLuLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        struct relu_param* param = (struct relu_param*)node->op.param_mem;

        float slop_param= param->negative_slope;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            // if(bForcedNHWCMode_ == true)    //
            // {
            //     TensorInfo* pClTensorInfo = itensor->info();
            //     if(pClTensorInfo->data_layout() == DataLayout::NCHW)
            //     {
            //         int* dim = input_tensor->dims;
            //         assert(input_tensor->dim_num == 4);
            //         pClTensorInfo->set_tensor_shape(TensorShape(dim[1], dim[3], dim[2], dim[0]));
            //         pClTensorInfo->set_data_layout(DataLayout::NHWC);
            //     }
            //     else
            //     {
            //         assert(pClTensorInfo->data_layout() == DataLayout::NHWC);
            //     }
            // }
            // else
            // {
            //     TensorInfo* pClTensorInfo = itensor->info();

            //     int* dim = input_tensor->dims;
            //     assert(input_tensor->dim_num == 4);
            //     pClTensorInfo->set_tensor_shape(TensorShape(dim[3], dim[2], dim[1], dim[0]));
            //     pClTensorInfo->set_data_layout(DataLayout::NCHW);
            //     itensor->allocator()->init(*pClTensorInfo);

            // }
        }
        else
        {
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        struct ir_tensor* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        int* dim_o = out_tensor->dims;
        name = out_tensor->name;
        CLTensor* otensor = new CLTensor();
        int TengineDataLayOut = out_tensor->layout;

        if(bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
        {
            // need to re init datalayout to nhwc
            TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[2], dim_o[3], dim_o[0]), 1, data_type_);
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
        CLActivationLayer* relu = new CLActivationLayer();
        if(slop_param==0)
        {
            relu->configure(itensor, otensor, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        }
        else
        {
            relu->configure(itensor, otensor, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU,slop_param));
        }
        
        functions_map_.push_back(relu);
        return true;
    }

    bool AddReLu6Layer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        struct ir_tensor* out_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        name = out_tensor->name;
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(*(itensor->info()));
        tensors_map_[name] = otensor;

        CLActivationLayer* relu = new CLActivationLayer();
        relu->configure(itensor, otensor,
                        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6));

        functions_map_.push_back(relu);

        return true;
    }

    bool AddResizeLayer(struct ir_node* node)
    {
#ifdef ACL_EXTENSTION
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        if(!itensor)
        {
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        /*output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        int* dim_w = o_tensor->dims;
        name = o_tensor->name;
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim_w[2], dim_w[3], dim_w[1], dim_w[0]), 1, data_type_));
        tensors_map_[name] = otensor;

        CLResizeLayer* resize = new CLResizeLayer();
        resize->configure(itensor, otensor, ResizeType::NEAREST);

        functions_map_.push_back(resize);

        return true;

#else
        return false;
#endif
    }

    bool AddSoftmaxLayer(struct ir_node* node)
    {
        struct ir_graph* graph = node->graph;
        struct ir_tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
        std::string name = input_tensor->name;
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            //TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }

        /*output */
        struct ir_tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
        name = o_tensor->name;

        TensorInfo* info = itensor->info();
        int size = info->dimension(0) * info->dimension(1) * info->dimension(2);
        TensorShape shape(size);
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(shape, 1, data_type_));
        tensors_map_[name] = otensor;
        if(info->dimension(0) == 1)
        {
            MYSoftmaxLayer* softmax = new MYSoftmaxLayer();
            softmax->configure(itensor, otensor, data_type_);
            functions_map_.push_back(softmax);
        }
        else
        {
            CLSoftmaxLayer* softmax = new CLSoftmaxLayer();
            softmax->configure(itensor, otensor);
            functions_map_.push_back(softmax);
        }

        return true;
    }

    CLTensor* GetCLTensor(std::string name)
    {
        return tensors_map_[name];
    }
};

#endif    // __ACL_GRAPH_HPP
