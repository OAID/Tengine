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
 * Copyright (c) 2019, Open AI Lab
 * Author: haoluo@openailab.com
 */

#include <vector>
#include <math.h>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "operator/layernormlstm.hpp"
#include "kernel/layernorm_lstm/ref_layernorm_lstm_kernel.h"

namespace TEngine {

namespace RefLayernormLSTMOps {

struct RefLayernormLSTM : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    void InitRegistry(void);

    lnlstm_param param_lnlstm;
    ref_lnlstm_t kernel_run;
    KernelRegistry<ref_lnlstm_t> kernel_registry;

    Tensor* input;
    Tensor* i2i_weights_tensor;
    Tensor* i2c_weights_tensor;
    Tensor* i2f_weights_tensor;
    Tensor* i2o_weights_tensor;
    Tensor* igate_bias_tensor;
    Tensor* cgate_bias_tensor;
    Tensor* fgate_bias_tensor;
    Tensor* ogate_bias_tensor;
    Tensor* r2i_weights_tensor;
    Tensor* r2c_weights_tensor;
    Tensor* r2f_weights_tensor;
    Tensor* r2o_weights_tensor;
    Tensor* c2i_weights_tensor;
    Tensor* c2f_weights_tensor;
    Tensor* c2o_weights_tensor;
    Tensor* projection_weights_tensor;
    Tensor* projection_bias_tensor;
    Tensor* iactivationstate_tensor;
    Tensor* icellstate_tensor;
    Tensor* ilayer_norm_coefficients_tensor;
    Tensor* flayer_norm_coefficients_tensor;
    Tensor* clayer_norm_coefficients_tensor;
    Tensor* olayer_norm_coefficients_tensor;
    
    float* input_gate_scratch;
    float* forget_gate_scratch;
    float* cell_scratch;
    float* output_gate_scratch;

    const float LayerNormEpsilon = 1e-8;

    RefLayernormLSTM(void)
    {
        input = nullptr;
        i2i_weights_tensor = nullptr;
        i2c_weights_tensor = nullptr;
        i2f_weights_tensor = nullptr;
        i2o_weights_tensor = nullptr;
        igate_bias_tensor  = nullptr;
        cgate_bias_tensor  = nullptr;
        fgate_bias_tensor  = nullptr;
        ogate_bias_tensor  = nullptr;
        r2i_weights_tensor = nullptr;
        r2c_weights_tensor = nullptr;
        r2f_weights_tensor = nullptr;
        r2o_weights_tensor = nullptr;
        c2i_weights_tensor = nullptr;
        c2f_weights_tensor = nullptr;
        c2o_weights_tensor = nullptr;
        projection_weights_tensor = nullptr;
        projection_bias_tensor    = nullptr;
        icellstate_tensor         = nullptr;
        iactivationstate_tensor   = nullptr;
        ilayer_norm_coefficients_tensor = nullptr;
        flayer_norm_coefficients_tensor = nullptr;
        clayer_norm_coefficients_tensor = nullptr;
        olayer_norm_coefficients_tensor = nullptr;
        input_gate_scratch  = nullptr;
        forget_gate_scratch = nullptr;
        cell_scratch        = nullptr;
        output_gate_scratch = nullptr;
        kernel_run = nullptr;

        InitRegistry();
    }
};

bool RefLayernormLSTM::Prerun(Node* node)
{
    int in_num = node->GetInputNum();

    for(int count = 0; count < in_num; count++)
    {
        Tensor* temptensor = node->GetInputTensor(count);
        const std::string& name = temptensor->GetName();
        if(name.find("input") != std::string::npos)
        {
            input =  temptensor;
        }
        if(name.find("i2i_weights") != std::string::npos)
        {
            i2i_weights_tensor = temptensor;
        }
        if(name.find("i2c_weights") != std::string::npos)
        {
            i2c_weights_tensor = temptensor;
        }
        if(name.find("i2f_weights") != std::string::npos)
        {
            i2f_weights_tensor = temptensor;
        }
        if(name.find("i2o_weights") != std::string::npos)
        {
            i2o_weights_tensor = temptensor;
        }
        if(name.find("r2i_weights") != std::string::npos)
        {
            r2i_weights_tensor = temptensor;
        }
        if(name.find("r2c_weights") != std::string::npos)
        {
            r2c_weights_tensor = temptensor;    
        }
        if(name.find("r2f_weights") != std::string::npos)
        {
            r2f_weights_tensor = temptensor;
        }
        if(name.find("r2o_weights") != std::string::npos)
        {
            r2o_weights_tensor = temptensor;
        }
        if(name.find("c2i_weights") != std::string::npos)
        {
            c2i_weights_tensor = temptensor;    
        }
        if(name.find("c2f_weights") != std::string::npos)
        {
            c2f_weights_tensor = temptensor;    
        }
        if(name.find("c2o_weights") != std::string::npos)
        {
            c2o_weights_tensor = temptensor;        
        }
        if(name.find("igate_bias") != std::string::npos)
        {
            igate_bias_tensor  = temptensor;
        }
        if(name.find("cgate_bias") != std::string::npos)
        {
            cgate_bias_tensor  = temptensor;
        }
        if(name.find("fgate_bias") != std::string::npos)
        {
            fgate_bias_tensor  = temptensor;     
        }
        if(name.find("ogate_bias") != std::string::npos)
        {
            ogate_bias_tensor  = temptensor;    
        }
        if(name.find("projection_weight") != std::string::npos)
        {
            projection_weights_tensor = temptensor;
        }
        if(name.find("projection_bias") != std::string::npos)
        {
            projection_bias_tensor    = temptensor;    
        }
        if(name.find("iactivationstateTensor") != std::string::npos)
        {
            iactivationstate_tensor   = temptensor;
        }
        if(name.find("icellstatetensor") != std::string::npos)
        {
            icellstate_tensor         = temptensor;    
        }
        if(name.find("ilayer_norm_coefficients") != std::string::npos)
        {
            ilayer_norm_coefficients_tensor = temptensor;    
        }
        if(name.find("flayer_norm_coefficients") != std::string::npos)
        {
            flayer_norm_coefficients_tensor = temptensor;
        }
        if(name.find("clayer_norm_coefficients") != std::string::npos)
        {
            clayer_norm_coefficients_tensor = temptensor;
        }
        if(name.find("olayer_norm_coefficients") != std::string::npos)
        {
            olayer_norm_coefficients_tensor = temptensor;
        }
    }
    int batch_size  = input->GetShape().Shape(0);
    int cell_size  = i2o_weights_tensor->GetShape().Shape(0);

    if(!(i2c_weights_tensor==nullptr))
    {
        input_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
    }
    forget_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
    cell_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));
    output_gate_scratch = (float*) malloc(batch_size * cell_size * sizeof(float));

    Tensor* input = node->GetInputTensor(0);
    auto i_quant = input->GetQuantParam();

    // int weight_out = weight->GetShape().Shape(0);
    // if(weight_out == param.out_number)
    //     param.need_trans = 0;
    // else
    //     param.need_trans = 1;

    Tensor* output = node->GetOutputTensor(0);
    auto o_quant = output->GetQuantParam();

    if(input->GetDataType() == TENGINE_DT_UINT8 || input->GetDataType() == TENGINE_DT_INT8)
    {
        if(i_quant->size() == 0 || o_quant->size() == 0)
        {
            std::cerr << "FC <UINT8> one quant is NONE: <" << i_quant->size() << ","
                      << o_quant->size() << "\n";
            return false;
        }
        param_lnlstm.scale[0] = (*i_quant)[0].scale;
        param_lnlstm.scale[1] = (*o_quant)[0].scale;
        param_lnlstm.zero_point[0] = (*i_quant)[0].zero_point;
        param_lnlstm.zero_point[1] = (*o_quant)[0].zero_point;
    }
    int layout = exec_attr->graph_layout;
    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);
        return false;
    }

    return true;
}

bool RefLayernormLSTM::Run(Node* node)
{
    if(kernel_run == nullptr)
        return false;
    LayerNormLSTM* layernormlstm_op = dynamic_cast<LayerNormLSTM*>(node->GetOp());
    LayerNormLSTMParam* param = layernormlstm_op->GetParam();

    Tensor* output = node->GetOutputTensor(0);

    int batch_size  = input->GetShape().Shape(0);
    int input_size = input->GetShape().Shape(input->GetShape().GetDim().size()-1);
    int cell_size  = i2o_weights_tensor->GetShape().Shape(0);
    int output_size = r2o_weights_tensor->GetShape().Shape(1);
    int output_true_size = output->GetShape().Shape(output->GetShape().GetDim().size()-1);
    int sequence_size= input->GetShape().Shape(1);
    const bool use_cifg = (i2i_weights_tensor == nullptr);
    const bool use_peephole = (c2o_weights_tensor != nullptr);


    float* total_input_data = (float*)get_tensor_mem(input);
    float* total_output_data = (float*)get_tensor_mem(output);

    const float* i2i_weights_data = nullptr;
    const float* i2c_weights_data = nullptr;
    const float* i2f_weights_data = nullptr;
    const float* i2o_weights_data = nullptr;
    const float* igate_bias_data  = nullptr;
    const float* cgate_bias_data  = nullptr;
    const float* fgate_bias_data  = nullptr;
    const float* ogate_bias_data  = nullptr;
    const float* r2i_weights_data = nullptr;
    const float* r2c_weights_data = nullptr;
    const float* r2f_weights_data = nullptr;
    const float* r2o_weights_data = nullptr;
    const float* c2i_weights_data = nullptr;
    const float* c2f_weights_data = nullptr;
    const float* c2o_weights_data = nullptr;
    const float* projection_weights_data = nullptr;
    const float* projection_bias_data    = nullptr;
    float* icellstate_data         = nullptr;
    float* iactivationstate_data   = nullptr;
    const float* ilayer_norm_coefficients_data = nullptr;
    const float* flayer_norm_coefficients_data = nullptr;
    const float* clayer_norm_coefficients_data = nullptr;
    const float* olayer_norm_coefficients_data = nullptr;

    if(!use_cifg)
    {
        i2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2i_weights_tensor));
        r2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2i_weights_tensor));
        igate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(igate_bias_tensor));
    }

    i2c_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2c_weights_tensor));
    i2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2f_weights_tensor));
    i2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(i2o_weights_tensor));
    cgate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(cgate_bias_tensor));
    fgate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(fgate_bias_tensor));
    ogate_bias_data  = const_cast<const float*>((float*)get_tensor_mem(ogate_bias_tensor));
    r2c_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2c_weights_tensor));
    r2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2f_weights_tensor));
    r2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(r2o_weights_tensor));
    icellstate_data  = (float*)get_tensor_mem(icellstate_tensor);
    iactivationstate_data = (float*)get_tensor_mem(iactivationstate_tensor);

    if(use_peephole)
    {
        if(!use_cifg)
        {
            c2i_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2i_weights_tensor));
        }
        c2f_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2f_weights_tensor));
        c2o_weights_data = const_cast<const float*>((float*)get_tensor_mem(c2o_weights_tensor));
    }

    if(projection_weights_tensor)
    {
        projection_weights_data = const_cast<const float*>((float*)get_tensor_mem(projection_weights_tensor));
    }
    if(projection_bias_tensor)
    {
        projection_bias_data = const_cast<const float*>((float*)get_tensor_mem(projection_bias_tensor));   
    }

    if(!use_cifg)
    {
        ilayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(ilayer_norm_coefficients_tensor));
    }
    clayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(clayer_norm_coefficients_tensor));
    olayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(olayer_norm_coefficients_tensor));
    flayer_norm_coefficients_data = const_cast<const float*>((float*)get_tensor_mem(flayer_norm_coefficients_tensor));
    
    
    param_lnlstm.batch_size=batch_size;
    param_lnlstm.input_size=input_size;
    param_lnlstm.cell_size=cell_size;
    param_lnlstm.output_size=output_size;
    param_lnlstm.output_true_size=output_true_size;
    param_lnlstm.sequence_size=sequence_size;
    param_lnlstm.i2i_weights_data=i2i_weights_data;
    param_lnlstm.i2c_weights_data=i2c_weights_data;
    param_lnlstm.i2f_weights_data=i2f_weights_data;
    param_lnlstm.i2o_weights_data=i2o_weights_data;
    param_lnlstm.igate_bias_data=igate_bias_data;
    param_lnlstm.cgate_bias_data=cgate_bias_data;
    param_lnlstm.fgate_bias_data=fgate_bias_data;
    param_lnlstm.ogate_bias_data=ogate_bias_data;
    param_lnlstm.r2i_weights_data=r2i_weights_data;
    param_lnlstm.r2c_weights_data=r2c_weights_data;
    param_lnlstm.r2f_weights_data=r2f_weights_data;
    param_lnlstm.r2o_weights_data=r2o_weights_data;
    param_lnlstm.c2i_weights_data=c2i_weights_data;
    param_lnlstm.c2f_weights_data=c2f_weights_data;
    param_lnlstm.c2o_weights_data=c2o_weights_data;
    param_lnlstm.projection_weights_data=projection_weights_data;
    param_lnlstm.projection_bias_data=projection_bias_data;
    param_lnlstm.icellstate_data=icellstate_data;
    param_lnlstm.iactivationstate_data=iactivationstate_data;
    param_lnlstm.ilayer_norm_coefficients_data=ilayer_norm_coefficients_data;
    param_lnlstm.flayer_norm_coefficients_data=flayer_norm_coefficients_data;
    param_lnlstm.clayer_norm_coefficients_data=clayer_norm_coefficients_data;
    param_lnlstm.olayer_norm_coefficients_data=olayer_norm_coefficients_data;
    param_lnlstm.input_gate_scratch=input_gate_scratch;
    param_lnlstm.forget_gate_scratch=forget_gate_scratch;
    param_lnlstm.cell_scratch=cell_scratch;
    param_lnlstm.output_gate_scratch=output_gate_scratch;
    param_lnlstm.fused_activation=param->fused_activation;
    param_lnlstm.cell_clip=param->cell_clip;
    param_lnlstm.proj_clip=param->proj_clip;


    // if(TENGINE_DT_INT8 == inputTensor->GetDataType())
    // {
    //     auto* out_quant = outputTensor->GetQuantParam();
    //     QuantParam q_param;
    //     q_param.scale = op_param.out_scale;
    //     q_param.zero_point = 0;
    //     out_quant->resize(0);
    //     out_quant->push_back(q_param);
    // }
    if(TENGINE_DT_UINT8 == input->GetDataType() || TENGINE_DT_INT8 == input->GetDataType())
    {
        auto* in_quant = input->GetQuantParam();
        if(in_quant->size())
        {
            param_lnlstm.scale[0]=(*in_quant)[0].scale;
            param_lnlstm.zero_point[0]=(*in_quant)[0].zero_point;
        }
    }
    if(TENGINE_DT_UINT8 == input->GetDataType())
    {
        auto* out_quant = output->GetQuantParam();
        if(out_quant->size())
        {
            param_lnlstm.scale[1] = (*out_quant)[0].scale;
            param_lnlstm.zero_point[1] = (*out_quant)[0].zero_point;
        }
    }

    if(kernel_run(total_input_data, total_output_data, &param_lnlstm) < 0)
        return false;

    
    if(TENGINE_DT_INT8 == input->GetDataType())
    {
        auto* out_quant = output->GetQuantParam();
        QuantParam q_param;
        q_param.scale = param_lnlstm.scale[1];
        q_param.zero_point = 0;
        out_quant->resize(0);
        out_quant->push_back(q_param);
    }
    return true;
}

void RefLayernormLSTM::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_lnlstm_t )ref_layernormlstm_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    RefLayernormLSTM* ops = new RefLayernormLSTM();

    LOG_DEBUG() << "Demo RefLayernormLSTMOpOp is selected\n";

    return ops;
}

}    // namespace RefLayernormLSTMOps

void RegisterRefLayernormLSTMOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "LayerNormLSTM", RefLayernormLSTMOps::SelectFunc, 1000);
}

}    // namespace TEngine
