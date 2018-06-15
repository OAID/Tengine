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
#include "operator/convolution.hpp"
#include<math.h>
#include <cblas.h>




namespace TEngine {

namespace ConvolutionImpl 
{


struct ConvolutionOps: public NodeOps 
{

    void im2col(float *data_img, float *data_col,
                 int inh, int inw, int inc,
                 int outh, int outw, int outc,
                 int ksize_h, int ksize_w, int sh, int sw,
                 int ph, int pw, int dh, int dw)
    {

        int channels_col = ksize_h*ksize_w*inc;
        for (int c = 0; c < channels_col; ++c) 
        {
            int kw = c % ksize_w;
            int c_ = c / ksize_w;
            int kh = c_ % ksize_h;
            c_ = c_ / ksize_h;
		    for (int h = 0; h < outh; ++h)
		    {
			    for (int w = 0; w < outw; ++w)
			    {
                    int im_row = kh * dh + h * sh - ph;
                    int im_col = kw * dw + w * sw - pw;
				    int col_index = (c * outh + h) * outw + w;
				    data_col[col_index] =
				        (im_row >= 0 && im_col >= 0 && im_row < inh && im_col < inw)?
				        data_img[im_col + inw*(im_row + inh*c_)] : 0;
			    }
		    }
	    }
    }
    void relu(float* data,int size)
    {
        for(int i=0;i<size;i++)
        {
            if(data[i]<0)
            {
                data[i]=0;
            }
        }
    }
    void add_bias(float *output, float *bias, int c_out, int hw)
    {
        for(int c = 0; c < c_out; ++c)
        {
            for(int i = 0; i < hw; ++i)
            {
                output[c*hw +i] += bias[c];
            }
        }
    }
    bool Prerun(Node * node)
    {
        Convolution *conv_op = dynamic_cast<Convolution *>(node->GetOp());
        ConvParam *param = conv_op->GetParam();

        const Tensor * input_tensor=node->GetInputTensor(0);
        const TShape&  shape=input_tensor->GetShape();
        const std::vector<int> in_dims=shape.GetDim();

         Tensor * output_tensor=node->GetOutputTensor(0);
         TShape&  shape1=output_tensor->GetShape();
         std::vector<int> out_dims=shape1.GetDim();

        int size=param->kernel_h*param->kernel_w *in_dims[1]/param->group*out_dims[2]*out_dims[3];
        float * buffer = (float*)std::malloc(sizeof(float) * size);
        (*node)["buffer"]=buffer;
        return true;
    }

    bool Run(Node *node)
    {
        bool relu_fused=false;
        if(node->ExistAttr("Fused.ReLu"))
        {
            relu_fused=true;
        }

        bool debug_conv=false;
        const char * debug_env=std::getenv("DEBUG_CONV");
        if((debug_env) && (debug_env[0]=='1'))
        {
            debug_conv=true;
        }

        Tensor *input_tensor = node->GetInputTensor(0);
        Tensor *output_tensor = node->GetOutputTensor(0);
        Tensor *weight_tensor = node->GetInputTensor(1);
        float * buffer  = any_cast<float *>(node->GetAttr("buffer"));

        Convolution *conv_op = dynamic_cast<Convolution *>(node->GetOp());
        ConvParam *param = conv_op->GetParam();

        float *input = (float *)get_tensor_mem(input_tensor);
        float *output = (float *)get_tensor_mem(output_tensor);
        float *kernel = (float *)get_tensor_mem(weight_tensor);

        const TShape &shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();
        const TShape &shape1 = output_tensor->GetShape();
        const std::vector<int> out_dims = shape1.GetDim();

        int batch_number = in_dims[0];
        int inc=in_dims[1];
        int inh=in_dims[2];
        int inw=in_dims[3];
        int in_chw= inc*inh*inw;

        int outc= out_dims[1];
        int outh= out_dims[2];
        int outw= out_dims[3];
        int out_hw = outh * outw;
        int out_chw = out_hw * outc;
        
        int ksize_h = param->kernel_h;
        int ksize_w = param->kernel_w;
        int pad_w = param->pads[1];  
        int pad_h = param->pads[0];  

        int stride_w    = param->stride_w;
        int stride_h    = param->stride_h;
        int dilation_w  = param->dilation_w;
        int dilation_h  = param->dilation_h;
        int group = param ->group;

        int inc_g= inc/group;
        int outc_g= outc/group;

        int m = outc_g;
        int k = ksize_h * ksize_w * inc_g;
        int n = outh * outw;
        
        int in_chw_g = inh * inw * inc_g;
        int out_chw_g = outc_g * out_hw;
        int kernel_size_g = outc_g * ksize_h *ksize_w * inc_g;

        bool have_biases  = (node->GetInputNum() > 2);
        float* biases = nullptr;

        if(debug_conv)
        {
            std::cout<<inc<<" "<<inh<<" "<<inw<<"\tksp dg: "
                    <<ksize_h<<" "<<stride_h<<" "<<pad_h<<" "<<dilation_w<<" "<<group<<"\t"
                    <<outc<<" "<<outh<<" "<<outw<<"\t";
        }

        
        for (int i = 0; i < batch_number; i++)
        {
            for(int g = 0; g < group; g++ )
            {
                im2col(input + i*in_chw+  g*in_chw_g, buffer,
                    inh,  inw,  inc_g,
                    outh, outw, outc_g,
                    ksize_h, ksize_w, stride_h, stride_w,
                    pad_h,pad_w,dilation_h,dilation_w);
               
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                            m, n, k, 1, kernel + g* kernel_size_g, k, buffer, n, 0, 
                            output + i* out_chw+ g* out_chw_g, n);
            }
        }        
        if(have_biases)
        {
            biases = (float *) get_tensor_mem(node->GetInputTensor(2));
            for(int i=0;i<batch_number;i++)
            {
                add_bias(output+i*out_chw, biases, outc, out_hw);
            }
        }
        if(relu_fused)
        {
            relu(output,batch_number*out_chw);
        }
        if(debug_conv)
        {
            std::cout<<output[0]<<" "<<output[10]<<"\n";
        }

        return true;
    }

    bool Postrun(Node * node)
    {
        float * addr;
        addr=any_cast<float *>(node->GetAttr("buffer"));
        std::free(addr);
        return true;
    }


};

} //namespace ConvolutionImpl


using namespace ConvolutionImpl;
void RegisterConvBlasNodeExec(void)
{
    ConvolutionOps *ops = new ConvolutionOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common",
                                                  "Convolution", ops);
}

} //namespace TEngine


