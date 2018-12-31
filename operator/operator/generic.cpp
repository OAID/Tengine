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
 * Author: haitao@openailab.com
 */
#include <string.h>

#include "operator/generic.hpp"
#include "custom_kernel.hpp"
#include "tengine_errno.hpp"

namespace TEngine {

bool Generic::InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout)
{
    if(!ExistAttr(ATTR_CUSTOM_KERNEL))
    {
        LOG_ERROR() << "generic node: no backend custom kernel set\n";
        set_tengine_errno(ENOENT);
        return false;
    }

    struct custom_kernel_ops* k_ops = any_cast<struct custom_kernel_ops*>(GetAttr(ATTR_CUSTOM_KERNEL));

    /* double check, this should not happen, as when set custom_kernel, there is a check */

    if(strncmp(k_ops->op, param_.op_name, strlen(k_ops->op)))
    {
        LOG_ERROR() << "generic node: OP mismatch: expect " << param_.op_name << " real " << k_ops->op << "\n";
        set_tengine_errno(EINVAL);
        return false;
    }

    /* check input/output number */

    int input_num = ishape.size();
    int output_num = oshape.size();

    if(input_num > param_.max_input_num)
    {
        LOG_ERROR() << "generic node: input number mismatch: max " << param_.max_input_num << " real" << input_num
                    << "\n";

        set_tengine_errno(EINVAL);
        return false;
    }

    if(output_num > param_.max_output_num)
    {
        LOG_ERROR() << "generic node: output number mismatch: max " << param_.max_output_num << " real" << output_num
                    << "\n";

        set_tengine_errno(EINVAL);
        return false;
    }

    if(k_ops->infer_shape == nullptr)
    {
        LOG_ERROR() << "generic node: customer has no infer shape method\n";

        set_tengine_errno(ENOTSUP);
        return false;
    }

    /* prepare parameters */

    int** inputs = ( int** )malloc(input_num * sizeof(int*));

    for(int i = 0; i < input_num; i++)
    {
        inputs[i] = ( int* )malloc(MAX_SHAPE_DIM_NUM * sizeof(int));

        for(int j = 0; j < MAX_SHAPE_DIM_NUM; j++)
            inputs[i][j] = 0;

        const TShape& shape = ishape[i];
        const std::vector<int> dims = shape.GetDim();

        for(unsigned int j = 0; j < dims.size(); j++)
            inputs[i][j] = dims[j];
    }

    int** outputs = ( int** )malloc(output_num * sizeof(int*));

    for(int i = 0; i < output_num; i++)
    {
        outputs[i] = ( int* )malloc(MAX_SHAPE_DIM_NUM * sizeof(int));

        for(int j = 0; j < MAX_SHAPE_DIM_NUM; j++)
            outputs[i][j] = 0;
    }

    int ret = k_ops->infer_shape(k_ops, ( const int** )inputs, input_num, outputs, output_num, layout);

    if(ret == 0)
    {
        for(int i = 0; i < output_num; i++)
        {
            TShape& shape = oshape[i];

            std::vector<int>& dims = shape.GetDim();

            int* o_dim = outputs[i];

            for(int j = 0; j < MAX_SHAPE_DIM_NUM; j++)
            {
                int idx = o_dim[j];

                if(idx == 0)
                    break;

                dims.push_back(idx);
            }
        }
    }

    /* free memory */

    for(int i = 0; i < input_num; i++)
        free(inputs[i]);

    free(inputs);

    for(int i = 0; i < output_num; i++)
        free(outputs[i]);

    free(outputs);

    if(ret < 0)
    {
        set_tengine_errno(EFAULT);
        return false;
    }

    return true;
}

void Generic::SetSchema(void)
{
    Input({"input:float32"})
        .Output({"output:float32"})
        .SetAttr("max_input_num", 1)
        .SetAttr("max_output_num", 1)
        .SetAttr("op_name", "none")
        .SetDoc(R"DOC(Generic Operator)DOC");
}

}    // namespace TEngine
