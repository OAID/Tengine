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
 * Author: lmzhang@openailab.com
 */
#include "operator/layernormlstm.hpp"
#include "operator/layernormlstm_param.hpp"
#include "static_graph.hpp"

namespace TEngine{

bool LayerNormLSTM::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    const TShape input_shape = ishape[0];

    int batch_size = input_shape.Shape(0);
    int sequence_size = input_shape.Shape(1);
    std::vector<int> dims(3);

    dims[0] = batch_size;
    dims[1] = sequence_size;
    dims[2] = param_.output_size;
    // std::cout<<dims[0]<<","<< dims[1]<<","<<dims[2]<<"\n"; 

    oshape[0].SetDim(dims);

    return true;    
}

void LayerNormLSTM::SetSchema(void)
{
    //i means input, c means cell, f means forget, o means output, r means recurrent
    Input({ "input:float32",
            "i2i_weights:float32", "i2f_weights:float32", "i2c_weights:float32", "i2o_weights:float32",
            "r2i_weights:float32", "r2f_weights:float32", "r2c_weights:float32", "r2o_weights:float32",
            "c2i_weights:float32", "c2f_weights:float32", "c2o_weights:float32",
            "igate_bias:float32", "cgate_bias:float32", "fgate_bias:float32", "ogate:float32",
            "projection_weights:float32","projection_bias:float32",
            "iactivationstateTensor:float32","icellstatetensor:float32",
            "ilayer_norm_coefficients:float32","flayer_norm_coefficients:float32",
            "clayer_norm_coefficients:float32","olayer_norm_coefficients:float32"})
    .Output({"output:float32"})
    .SetAttr("hidden_size", 0)
    .SetAttr("output_size", 0)
    .SetAttr("cell_clip", 0.0f)
    .SetAttr("proj_clip", 0.0f)
    .SetAttr("kernel_type", KernelType::kFullKernel)
    .SetAttr("fused_activation", FusedActivation::kNone)
    .SetDoc(R"DOC(LayerNormLSTM Cell
              input: input sequences, a 2D tensor [batch_size, seq_size*input_size]
              output: output sequences, a 2D tensor [batch_size, seq_size*output_size]
                 )DOC");

}

}
