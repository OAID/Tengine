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
 * Author: zpluo@openailab.com
 */

#include <vector>

#include "data_type.hpp"
#include "kernel_registry.hpp"
#include "tengine_errno.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "operator/tile.hpp"
#include "kernel/tile/ref_tile_kernel.h"

namespace TEngine {

namespace RefTileOps {

struct TileOps : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    void InitRegistry(void);

    // struct tile_param op_param;
    ref_tile_t kernel_run;

    KernelRegistry<ref_tile_t> kernel_registry;

    TileOps(void)
    {
        kernel_run = nullptr;

        InitRegistry();
    }
};
bool TileOps::Prerun(Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    int layout = exec_attr->graph_layout;

    if(!kernel_registry.GetKernel(kernel_run, layout, input->GetDataType()))
    {
        set_tengine_errno(ENOENT);

        return false;
    }

    return true;
}

bool TileOps::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Tensor* output_tensor = node->GetOutputTensor(0);
    const TShape& shape = input_tensor->GetShape();

    const TShape& shape1 = output_tensor->GetShape();

    float* input = ( float* )get_tensor_mem(input_tensor);
    float* output = ( float* )get_tensor_mem(output_tensor);

    Tile* tile_op = dynamic_cast<Tile*>(node->GetOp());
    TileParam* param = tile_op->GetParam();
    std::vector<int> repeat = param->reps;
    int size = repeat.size();

    for(int i = 0; i < 4 - size; i++)
    {
        repeat.push_back(1);
    }
    std::vector<int> inDim = shape.GetDim();
    std::vector<int> outDim = shape1.GetDim();

    float scale = 1.f;
    int zero_point = 0;
    if(input_tensor->GetDataType() == TENGINE_DT_INT8 || input_tensor->GetDataType() == TENGINE_DT_UINT8)
    {
        auto quant_param = input_tensor->GetQuantParam();
        scale = (*quant_param)[0].scale;
        zero_point = (*quant_param)[0].zero_point;
        auto out_quant_param = output_tensor->GetQuantParam();
        out_quant_param->resize(0);
        out_quant_param->push_back((*quant_param)[0]);
    }

    int ret = -1;

    ret = kernel_run(input, output, repeat, inDim, outDim, scale, zero_point);

    if(ret < 0)
        return false;
    else
        return true;
}

bool TileOps::Postrun(Node* node)
{
    return true;
}

void TileOps::InitRegistry(void)
{
#ifdef CONFIG_KERNEL_FP32
    kernel_registry.Register(( ref_tile_t )ref_tile_fp32, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP32);
    kernel_registry.Register(( ref_tile_t )ref_tile_fp32, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP32);
#endif

#ifdef CONFIG_KERNEL_FP16
    kernel_registry.Register(( ref_tile_t )ref_tile_fp16, TENGINE_LAYOUT_NCHW, TENGINE_DT_FP16);
    kernel_registry.Register(( ref_tile_t )ref_tile_fp16, TENGINE_LAYOUT_NHWC, TENGINE_DT_FP16);
#endif

#ifdef CONFIG_KERNEL_INT8
    kernel_registry.Register(( ref_tile_t )ref_tile_int8, TENGINE_LAYOUT_NCHW, TENGINE_DT_INT8);
    kernel_registry.Register(( ref_tile_t )ref_tile_int8, TENGINE_LAYOUT_NHWC, TENGINE_DT_INT8);
#endif

#ifdef CONFIG_KERNEL_UINT8
    kernel_registry.Register(( ref_tile_t )ref_tile_uint8, TENGINE_LAYOUT_NCHW, TENGINE_DT_UINT8);
    kernel_registry.Register(( ref_tile_t )ref_tile_uint8, TENGINE_LAYOUT_NHWC, TENGINE_DT_UINT8);
#endif
}

NodeOps* SelectFunc(const CPUInfo* info, Node* node)
{
    TileOps* ops = new TileOps();

    LOG_DEBUG() << "TileOps RefOp is selected\n";

    return ops;
}

}    // namespace RefTileOps
void RegisterRefTileOps(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME, "Tile", RefTileOps::SelectFunc, 1000);
}
}    // namespace TEngine
