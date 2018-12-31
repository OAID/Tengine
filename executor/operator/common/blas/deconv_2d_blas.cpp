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
#include "operator/deconvolution.hpp"
#include <math.h>
#include <sys/time.h>
#include <cblas.h>

namespace TEngine {

namespace DeconvolutionImpl {

struct DeconvBlasOps : public NodeOps
{
    bool Prerun(Node* node)
    {
        // param
        Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
        DeconvParam* param_ = deconv_op->GetParam();

        const Tensor* input_tensor = node->GetInputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> dims = shape.GetDim();

        int size = dims[2] * dims[3] * param_->kernel_size * param_->kernel_size * param_->num_output;
        float* buffer = ( float* )std::malloc(sizeof(float) * size);
        memset(buffer, 0, size * sizeof(float));
        (*node)["buffer"] = buffer;

        return true;
    }

    void add_bias(float* output, float* bias, int c_out, int hw)
    {
        float* out_ptr = output;
        for(int c = 0; c < c_out; ++c)
        {
            float val = bias[c];
            for(int i = 0; i < hw; ++i)
            {
                *out_ptr += val;
                out_ptr++;
            }
        }
    }
    void gemm_tn(int M, int N, int K, float* A, int lda, float* B, int ldb, float* C, int ldc)
    {
        int i, j, k;
        for(i = 0; i < M; ++i)
        {
            for(k = 0; k < K; ++k)
            {
                register float temp_a = A[k * lda + i];
                for(j = 0; j < N; ++j)
                {
                    C[i * ldc + j] += temp_a * B[k * ldb + j];
                }
            }
        }
    }

    void col2im(float* data_col, float* data_im, int channels, int height, int width, int ksize, int stride, int pad,
                int dilation, int h_out, int w_out)
    {
        int c, h;
        float* out = data_col;
        for(c = 0; c < channels; ++c)
        {
            for(int ki = 0; ki < ksize; ki++)
            {
                for(int kj = 0; kj < ksize; kj++)
                {
                    const int im_col = kj * dilation - pad;
                    const int w_low = std::max(0, -im_col / stride + (-im_col % stride > 0));
                    const int w_high = std::min(w_out, (width - im_col) / stride + ((width - im_col) % stride > 0));

                    for(h = 0; h < h_out; ++h)
                    {
                        int im_row = ki * dilation + h * stride - pad;
                        if(im_row < 0 || im_row >= height)
                        {
                            out += w_out;
                        }
                        else
                        {
                            float* dst = data_im + width * (im_row + height * c) + im_col + w_low * stride;
                            const float* end = out + w_high;

                            out += w_low;
                            while(out < end)
                            {
                                *dst += *(out++);
                                dst += stride;
                            }
                            out += w_out - w_high;
                        }
                    }
                }
            }
        }
    }

    bool Run(Node* node)    //
    {
        // input
        const Tensor* input_tensor = node->GetInputTensor(0);
        float* input = ( float* )get_tensor_mem(input_tensor);
        const TShape& in_shape = input_tensor->GetShape();
        const std::vector<int> in_dims = in_shape.GetDim();

        // output
        Tensor* output_tensor = node->GetOutputTensor(0);
        float* output = ( float* )get_tensor_mem(output_tensor);
        const TShape& out_shape = output_tensor->GetShape();
        const std::vector<int> out_dims = out_shape.GetDim();

        // weight
        const Tensor* weight_tensor = node->GetInputTensor(1);
        float* weight = ( float* )get_tensor_mem(weight_tensor);

        // bias
        const Tensor* bias_tensor = node->GetInputTensor(2);
        float* bias = ( float* )get_tensor_mem(bias_tensor);

        // param
        Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
        DeconvParam* param_ = deconv_op->GetParam();
        int pad = param_->pad;
        int stride = param_->stride;
        int ksize = param_->kernel_size;
        int dilation = param_->dilation;

        // buffer
        float* buffer = any_cast<float*>(node->GetAttr("buffer"));

        // shape
        int batch = in_dims[0];
        int chw_in = in_dims[1] * in_dims[2] * in_dims[3];
        int c_in = in_dims[1];
        int h_in = in_dims[2];
        int w_in = in_dims[3];
        int c_out = out_dims[1];
        int h_out = out_dims[2];
        int w_out = out_dims[3];
        int chw_out = c_out * h_out * w_out;
        int hw_out = out_dims[2] * out_dims[3];
        int out_size = out_dims[0] * chw_out;

        memset(output, 0, out_size * sizeof(float));
        int m = ksize * ksize * c_out;
        int n = h_in * w_in;
        int k = c_in;

        for(int b = 0; b < batch; ++b)
        {
            float* inp = input + b * chw_in;
            float* out_ptr = output + b * chw_out;

            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1, weight, m, inp, n, 0, buffer, n);

            col2im(buffer, out_ptr, c_out, h_out, w_out, ksize, stride, pad, dilation, h_in, w_in);

            add_bias(out_ptr, bias, c_out, hw_out);
        }

        return true;
    }

    bool Postrun(Node* node)
    {
        float* addr;

        addr = any_cast<float*>(node->GetAttr("buffer"));
        std::free(addr);
        return true;
    }
};

}    // namespace DeconvolutionImpl

using namespace DeconvolutionImpl;

void RegisterDeconvBlasNodeExec(void)
{
    DeconvBlasOps* ops = new DeconvBlasOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "Deconvolution", ops);
}

}    // namespace TEngine
