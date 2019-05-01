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
 * Author: haoluo@openailab.com
 */
#ifndef __ACL_GRAPH_HPP
#define __ACL_GRAPH_HPP

#include <array>
#include <random>
#include <string>
#include <vector>
#include <arm_neon.h>

#include "graph.hpp"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

#include "tensor_mem.hpp"
#include "operator/convolution.hpp"
#include "operator/pooling.hpp"
#include "operator/batch_norm.hpp"
#include "operator/eltwise.hpp"

using namespace arm_compute;

#define USE_CPU_CONVERT
//#define ACL_EXTENSTION

namespace TEngine {

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
            float16_t* src = reinterpret_cast<float16_t*>(input_org->buffer());
            float16_t* dst = reinterpret_cast<float16_t*>(_input.buffer());
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

    CLGraph(std::string name, DataType type)
    {
        name_ = name;
        data_type_ = type;
    };

    ~CLGraph() {}

    void Run(void)
    {
        int size = functions_map_.size();
        for(int i = 0; i < size; i++)
        {
            functions_map_[i]->run();
        }
    }

    bool AddInputLayer(Node* node)
    {
        /* output */
        Tensor* tensor = node->GetOutputTensor(0);
        std::string name = tensor->GetName();
        std::vector<int> dim_w = tensor->GetShape().GetDim();

        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim_w[2], dim_w[3], dim_w[1], dim_w[0]), 1, data_type_));

        tensors_map_[name] = otensor;

        return true;
    }

    bool AddBNLayer(Node* node, Node* node_scale)
    {
        BatchNorm* bn_op = dynamic_cast<BatchNorm*>(node->GetOp());
        BatchNormParam* param = bn_op->GetParam();
        float eps = param->eps;

        /* input */
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        int channel = input_tensor->GetShape().GetC();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        /* gamma */
        Tensor* gamma_tensor = node_scale->GetInputTensor(1);
        CLTensor* gtensor = nullptr;
        if(gamma_tensor)
        {
            name = gamma_tensor->GetName();
            gtensor = new CLTensor();
            gtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = gtensor;
        }
        /* beta */
        Tensor* beta_tensor = node_scale->GetInputTensor(2);
        CLTensor* btensor = nullptr;
        if(beta_tensor)
        {
            name = beta_tensor->GetName();
            btensor = new CLTensor();
            btensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = btensor;
        }

        /* means */
        Tensor* means_tensor = node->GetInputTensor(3);
        name = means_tensor->GetName();
        CLTensor* mtensor = new CLTensor();
        mtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
        tensors_map_[name] = mtensor;

        /* var */
        Tensor* var_tensor = node->GetInputTensor(4);
        CLTensor* vtensor = nullptr;
        if(var_tensor)
        {
            name = var_tensor->GetName();
            vtensor = new CLTensor();
            vtensor->allocator()->init(TensorInfo(TensorShape(channel), 1, data_type_));
            tensors_map_[name] = vtensor;
        }
        /* output */
        Tensor* out_tensor = node_scale->GetOutputTensor(0);
        name = out_tensor->GetName();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(*(itensor->info()));
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
        void* means = get_tensor_mem(means_tensor);
        void* vars = get_tensor_mem(var_tensor);
        /*for(int i=0;i<channel;i++)
        {
            vars_data[i] = 1.f/std::sqrt(vars[i]*rescale + eps);
            means_data[i] = -means[i]*rescale*vars_data[i];
        }*/

        copy_buffer(means_data, means, channel * 4, data_type_, DataType::F32);
        copy_buffer(vars_data, vars, channel * 4, data_type_, DataType::F32);

        mtensor->unmap();
        vtensor->unmap();

        if(btensor)
        {
            btensor->allocator()->allocate();
            btensor->map();
            void* beta_data = btensor->buffer();
            void* beta = get_tensor_mem(beta_tensor);
            copy_buffer(beta_data, beta, channel * 4, data_type_, DataType::F32);
            btensor->unmap();
        }
        if(gtensor)
        {
            gtensor->allocator()->allocate();
            gtensor->map();
            void* gamma_data = gtensor->buffer();
            void* gamma = get_tensor_mem(gamma_tensor);
            copy_buffer(gamma_data, gamma, channel * 4, data_type_, DataType::F32);
            gtensor->unmap();
        }

        return true;
    }

    bool AddConcatLayer(Node* node)
    {
        std::vector<ICLTensor*> inputs_vector;
        for(unsigned int i = 0; i < node->GetInputNum(); i++)
        {
            Tensor* tensor = node->GetInputTensor(i);
            std::string name = tensor->GetName();
            CLTensor* itensor = nullptr;

            if(tensors_map_.count(name))
            {
                itensor = tensors_map_[name];
            }
            else
            {
                LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
                return false;
            }
            inputs_vector.push_back(itensor);
        }

        /*output */
        Tensor* out = node->GetOutputTensor(0);
        std::vector<int> dim = out->GetShape().GetDim();
        std::string name = out->GetName();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim[3], dim[2], dim[1], dim[0]), 1, data_type_));
        tensors_map_[name] = otensor;

        CLDepthConcatenateLayer* concat = new CLDepthConcatenateLayer();

        /* 18.05 only support depth/channel concat */
        concat->configure(inputs_vector, otensor);
        functions_map_.push_back(concat);

        return true;
    }

    bool AddConvolutionLayer(Node* node)
    {
        void* acl_data = nullptr;
        void* data = nullptr;
        ActivationLayerInfo act_info;
        if(node->ExistAttr("Fused.ReLu"))
            act_info = ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU);
        if(node->ExistAttr("Fused.ReLu6"))
            act_info = ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU);
        Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
        ConvParam* param = conv_op->GetParam();

        int pad_x = param->pad_w0;
        int pad_y = param->pad_h0;
        int pad_x_1 = param->pad_w1;
        int pad_y_1 = param->pad_h1;
        int stride_x = param->stride_w;
        int stride_y = param->stride_h;
        int group = param->group;
        int outchan = param->output_channel;

        /* input */
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_DEBUG() << "Can't find node [" << node->GetName() << "] tensor named :" << name << "\n";
            return false;
        }
        /* weight */
        Tensor* w_tensor = node->GetInputTensor(1);
        std::vector<int> dim_w = w_tensor->GetShape().GetDim();
        name = w_tensor->GetName();
        CLTensor* wtensor = new CLTensor();
        wtensor->allocator()->init(TensorInfo(TensorShape(dim_w[2], dim_w[3], dim_w[1], dim_w[0]), 1, data_type_));
        tensors_map_[name] = wtensor;

        /* bias */
        Tensor* b_tensor = node->GetInputTensor(2);
        CLTensor* btensor = nullptr;
        if(b_tensor)
        {
            int channel = b_tensor->GetShape().GetSize();

            name = b_tensor->GetName();
            btensor = new CLTensor();
            btensor->allocator()->init(TensorInfo(TensorShape(channel, 1, 1, 1), 1, data_type_));
            tensors_map_[name] = btensor;
        }

        /* output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        std::vector<int> dim_o = o_tensor->GetShape().GetDim();
        name = o_tensor->GetName();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim_o[2], dim_o[3], dim_o[1], dim_o[0]), 1, data_type_));
        tensors_map_[name] = otensor;

        /* configure */
        if(group > 1 && group == outchan)
        {
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
            CLConvolutionLayer* clconv = new CLConvolutionLayer();
            clconv->configure(
                itensor, wtensor, btensor, otensor,
                PadStrideInfo(stride_x, stride_y, pad_x, pad_x_1, pad_y, pad_y_1, DimensionRoundingType::FLOOR),
                WeightsInfo(), Size2D(1U, 1U), act_info);
            functions_map_.push_back(clconv);
        }
        wtensor->allocator()->allocate();
        wtensor->map();
        data = get_tensor_mem(w_tensor);
        acl_data = wtensor->buffer();
        int size = w_tensor->GetTotalSize();
        copy_buffer(acl_data, data, size, data_type_, DataType::F32);
        wtensor->unmap();
        if(btensor)
        {
            btensor->allocator()->allocate();
            btensor->map();
            data = get_tensor_mem(b_tensor);
            acl_data = btensor->buffer();
            int size = b_tensor->GetTotalSize();
            copy_buffer(acl_data, data, size, data_type_, DataType::F32);
            btensor->unmap();
        }

        return true;
    }

    bool AddDropoutLayer(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        /*output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        name = o_tensor->GetName();
        tensors_map_[name] = itensor;

        return true;
    }

    bool AddEltwiseLayer(Node* node)
    {
        Tensor* input_tensor0 = node->GetInputTensor(0);
        std::string name = input_tensor0->GetName();
        CLTensor* itensor0 = nullptr;
        if(tensors_map_.count(name))
        {
            itensor0 = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }
        Tensor* input_tensor1 = node->GetInputTensor(1);
        name = input_tensor1->GetName();
        CLTensor* itensor1 = nullptr;
        if(tensors_map_.count(name))
        {
            itensor1 = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }
        /*output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        name = o_tensor->GetName();
        std::vector<int> dim = o_tensor->GetShape().GetDim();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim[2], dim[3], dim[1], dim[0]), 1, data_type_));
        tensors_map_[name] = otensor;

        Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
        EltwiseParam* param = eltwise_op->GetParam();
        if(ELT_SUM == param->type)
        {
            CLArithmeticAddition* add = new CLArithmeticAddition();
            add->configure(itensor0, itensor1, otensor, ConvertPolicy::WRAP);
            functions_map_.push_back(add);
        }
        else
        {
            printf("eltwise only support ADD!~~\n");
            return false;
        }

        return true;
    }

    bool AddFCLayer(Node* node)
    {
        /* Input */
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        if(!itensor)
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }
        /* weight */
        Tensor* w_tensor = node->GetInputTensor(1);
        name = w_tensor->GetName();
        int M = w_tensor->GetShape().GetN();
        int K = w_tensor->GetShape().GetC();
        CLTensor* wtensor = new CLTensor();
        wtensor->allocator()->init(TensorInfo(TensorShape(K, M), 1, data_type_));
        tensors_map_[name] = wtensor;
        /* bias */
        Tensor* b_tensor = node->GetInputTensor(2);
        CLTensor* btensor = nullptr;

        if(b_tensor)
        {
            name = b_tensor->GetName();
            btensor = new CLTensor();
            btensor->allocator()->init(TensorInfo(TensorShape(M), 1, data_type_));
            tensors_map_[name] = btensor;
        }

        /*output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        name = o_tensor->GetName();
        std::vector<int> dim_w = o_tensor->GetShape().GetDim();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(dim_w[1]), 1, data_type_));
        tensors_map_[name] = otensor;

        /* FC Layer */
        bool transpose_w = (dim_w[1] == M) ? true : false;
        CLFullyConnectedLayer* fc = new CLFullyConnectedLayer();

        fc->configure(itensor, wtensor, btensor, otensor, transpose_w);
        functions_map_.push_back(fc);
        wtensor->allocator()->allocate();
        wtensor->map();
        void* data = get_tensor_mem(w_tensor);
        void* acl_data = wtensor->buffer();
        int size = w_tensor->GetTotalSize();
        copy_buffer(acl_data, data, size, data_type_, DataType::F32);
        wtensor->unmap();
        if(btensor)
        {
            btensor->allocator()->allocate();
            btensor->map();
            data = get_tensor_mem(b_tensor);
            acl_data = btensor->buffer();
            int size = b_tensor->GetTotalSize();
            copy_buffer(acl_data, data, size, data_type_, DataType::F32);
            btensor->unmap();
        }
        return true;
    }

    bool AddPoolingLayer(Node* node)
    {
        Pooling* pool_op = dynamic_cast<Pooling*>(node->GetOp());
        PoolParam* param = pool_op->GetParam();
        int pad_x = param->pad_w0;
        int pad_y = param->pad_h0;
        int stride_x = param->stride_w;
        int stride_y = param->stride_h;
        int kernel_w = param->kernel_w ;
        int kernel_h = param->kernel_h;
        int type = param->alg;
        int global = param->global;

        Tensor* input_tensor = node->GetInputTensor(0);
        int channel = input_tensor->GetShape().GetC();
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        /* output */
        Tensor* o_tensor = node->GetOutputTensor(0);

        TensorInfo* info = itensor->info();
        int out_h = std::ceil(( float )(info->dimension(0) - kernel_h + 2 * pad_y) / stride_y) + 1;
        int out_w = std::ceil(( float )(info->dimension(1) - kernel_w + 2 * pad_x) / stride_x) + 1;
        name = o_tensor->GetName();

        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(TensorInfo(TensorShape(out_h, out_w, channel, 1), 1, data_type_));
        tensors_map_[name] = otensor;

        CLPoolingLayer* pooling = new CLPoolingLayer();
        PoolingLayerInfo pooling_info;
        if(global)
            pooling_info = PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX);
        else
            pooling_info =
                PoolingLayerInfo(type ? PoolingType::AVG : PoolingType::MAX, Size2D(kernel_w, kernel_h),
                                 PadStrideInfo(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::CEIL));

        pooling->configure(itensor, otensor, pooling_info);

        functions_map_.push_back(pooling);

        return true;
    }

    bool AddReLuLayer(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        Tensor* out_tensor = node->GetOutputTensor(0);
        name = out_tensor->GetName();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(*(itensor->info()));
        tensors_map_[name] = otensor;

        CLActivationLayer* relu = new CLActivationLayer();
        relu->configure(itensor, otensor, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        functions_map_.push_back(relu);

        return true;
    }

    bool AddReLu6Layer(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        Tensor* out_tensor = node->GetOutputTensor(0);
        name = out_tensor->GetName();
        CLTensor* otensor = new CLTensor();
        otensor->allocator()->init(*(itensor->info()));
        tensors_map_[name] = otensor;

        CLActivationLayer* relu = new CLActivationLayer();
        relu->configure(itensor, otensor,
                        ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6));

        functions_map_.push_back(relu);

        return true;
    }

    bool AddResizeLayer(Node* node)
    {
#ifdef ACL_EXTENSTION
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        if(!itensor)
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "]tensor named :" << name << "\n";
            return false;
        }

        /*output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        std::vector<int> dim_w = o_tensor->GetShape().GetDim();
        name = o_tensor->GetName();
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

    bool AddSoftmaxLayer(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        std::string name = input_tensor->GetName();
        CLTensor* itensor = nullptr;
        if(tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
        }
        else
        {
            LOG_INFO() << "can't find node [" << node->GetName() << "] tensor named :" << name << "\n";
            return false;
        }

        /*output */
        Tensor* o_tensor = node->GetOutputTensor(0);
        name = o_tensor->GetName();

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
}    // namespace TEngine

#endif    // __ACL_GRAPH_HPP
