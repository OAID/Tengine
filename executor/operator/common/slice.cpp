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
 * Author: ruizhang@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/slice.hpp"


namespace TEngine {

namespace SliceImpl {
const int default_prio = 200;
struct SliceOps : public NodeOps
{
    template <typename T>
    bool caffe_run(Node *node)
    {
        // get the slice param
        Slice * slice_op = dynamic_cast<Slice*>(node->GetOp());
        SliceParam * param = slice_op->GetParam();
        int slice_axis = param->axis;
        int num_slices = 1;
        int slice_size = 1;
        Tensor* input_tensor = node->GetInputTensor(0);

        const TShape& input_shape = input_tensor->GetShape();
        T* input = ( T* )get_tensor_mem(input_tensor);
        std::vector<int> in_dim = input_shape.GetDim();
        for(int i = 0; i < slice_axis; i++)
        {
            num_slices = num_slices * in_dim[i];
        }
        for(unsigned int i = slice_axis + 1; i < in_dim.size(); i++)
        {
            slice_size = slice_size * in_dim[i];
        }
        int in_slice = in_dim[slice_axis];
        int slice_index = 0;
        unsigned int out_num = node->GetOutputNum();
        for(unsigned int i = 0; i < out_num; i++)
        {
              Tensor* output_tensor = node->GetOutputTensor(i);
              T* output = (T* )get_tensor_mem(output_tensor);
              int out_slice = (output_tensor->GetShape()).Shape(slice_axis);

              for(int n = 0; n < num_slices; n++)
              {
                   int in_offset = (n * in_slice + slice_index) * slice_size;
                   int out_offset  = n * out_slice * slice_size;
                   memcpy(output+out_offset,input + in_offset,slice_size * out_slice * sizeof(T));
              }
              slice_index += out_slice;
        }
        return true;

    }
    template<typename T>
    bool tf_run(Node *node)
    {
        // get the slice param
        Slice * slice_op = dynamic_cast<Slice*>(node->GetOp());
        SliceParam * param = slice_op->GetParam();
        // get the input data
        Tensor* input_tensor = node->GetInputTensor(0);
        const TShape& input_shape = input_tensor->GetShape();
        T* input = (T* )get_tensor_mem(input_tensor);
        Tensor* output_tensor = node->GetOutputTensor(0);
        T *output = (T* )get_tensor_mem(output_tensor);
        std::vector<int> in_dim = input_shape.GetDim();
        int in_dim_new[4];
        int maxdim = 4;
        int begins[4];
        int sizes[4];
        int real_dim = param->begin_.size();
        int dim_idx = 0;
        for(int idx = 0; idx < maxdim; idx++)
        {
            if(maxdim - idx > real_dim)
            {
                begins[idx] = 0;
                sizes[idx] = 1;
                in_dim_new[idx] = 1;
            }
            else
            {
                begins[idx] = param->begin_[dim_idx];
                sizes[idx] = param->size_[dim_idx];
                in_dim_new[idx] = in_dim[dim_idx];
                dim_idx++;
            }
        }
        int in_dim_0 = in_dim_new[0];
        int in_dim_1 = in_dim_new[1];
        int in_dim_2 = in_dim_new[2];
        int in_dim_3 = in_dim_new[3];

        int start_dim_0 = (4 - real_dim) > 0 ? 0 : begins[0];
        int stop_dim_0 = ((4 - real_dim) > 0 || sizes[0] == -1)
                     ? in_dim_0 - start_dim_0
                     : start_dim_0 + sizes[0];
        int start_dim_1 = (3 - real_dim) > 0 ? 0 : begins[1];
        int stop_dim_1 = ((3 - real_dim) > 0 || sizes[1] == -1)
                     ? in_dim_1 - start_dim_1
                     : start_dim_1 + sizes[1];
        int start_dim_2 = (2 - real_dim) > 0 ? 0 : begins[2];
        int stop_dim_2 = ((2 - real_dim) > 0 || sizes[2] == -1)
                     ? in_dim_2 - start_dim_2
                     : start_dim_2 + sizes[2];
        int start_dim_3 = (1 - real_dim) > 0 ? 0 : begins[3];
        int stop_dim_3 = ((1 - real_dim) > 0 || sizes[3] == -1)
                     ? in_dim_3 - start_dim_3
                     : start_dim_3 + sizes[3];

        for(int n = start_dim_0; n < stop_dim_0;++n)
        {
            for(int i = start_dim_1; i < stop_dim_1; ++i)
            {
                for(int j = start_dim_2; j < stop_dim_2; ++j)
                {
                    int len = stop_dim_3 - start_dim_3;
                    int input_off = n * in_dim_1 * in_dim_2 * in_dim_3 +
                                    i * in_dim_2 * in_dim_3 +
                                    j * in_dim_3 + start_dim_3;
                    memcpy(output,input + input_off,len * sizeof(T));
                    output += len;
                }
            }
        }
        return true;
    }
    bool Run(Node* node)
    {
        Slice * slice_op = dynamic_cast<Slice*>(node->GetOp());
        SliceParam * param = slice_op->GetParam();
        if(param->iscaffe)
        {
            return caffe_run<float>(node);
        }
        else
        {
            return tf_run<float>(node);
        }
    }
};
NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
#ifdef CONFIG_ATUH_DEVICE
    if(!get_auth_float_enabled())
       return nullptr;
#endif

    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 ||exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;
    SliceOps* ops = new SliceOps();
    return ops;
}

}    // namespace SliceImpl

using namespace SliceImpl;

void RegisterSliceNodeExec(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("common", "Slice", SliceImpl::SelectFunc,
                                                  SliceImpl::default_prio))
        LOG_ERROR()<<__FUNCTION__<<" :Regist OP failed for prio["<<SliceImpl::default_prio<<"]\n";
}

}    // namespace TEngine
