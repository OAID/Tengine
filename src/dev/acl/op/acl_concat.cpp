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
 * Author: hhchen@openailab.com
 */


#include "acl_executor.hpp"

extern "C"
{
#include "tengine_op.h"
#include "concat_param.h"
}


bool CLGraph::AddConcatLayer(struct ir_node* node)
{
    fprintf(stderr, "Tengine ACl: Support OP(%d) OP_CONCAT.\n", node->idx);
    struct concat_param* param = ( struct concat_param* )node->op.param_mem;

    struct ir_graph* graph = node->graph;
    std::vector<ICLTensor*> inputs_vector;
    for (unsigned int i = 0; i < node->input_num; i++)
    {
        struct ir_tensor* tensor = get_ir_graph_tensor(graph, node->input_tensors[i]);
        char* name = tensor->name;
        CLTensor* itensor = nullptr;
        if (tensors_map_.count(name))
        {
            itensor = tensors_map_[name];
            if (bForcedNHWCMode_ == true)    //
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if (pClTensorInfo->data_layout() == DataLayout::NCHW)
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
            else
            {
                TensorInfo* pClTensorInfo = itensor->info();
                if (pClTensorInfo->data_layout() == DataLayout::NCHW)
                {
                    int* dim = tensor->dims;
                    assert(tensor->dim_num == 4);

                    pClTensorInfo->set_tensor_shape(TensorShape(dim[3], dim[2], dim[1], dim[0]));
                    //                    pClTensorInfo->set_data_layout(DataLayout::NHWC);
                }
                else
                {
                }
            }
        }
        else
        {
            // TLOG_ERR("can't find node [%s]tensor named :%s\n", node->name, name);
            return false;
        }
        inputs_vector.push_back(itensor);
        //        fprintf(stderr," in dimension0 %d \n",itensor->info()->dimension(0));
    }

    /*output */
    struct ir_tensor* out = get_ir_graph_tensor(graph, node->output_tensors[0]);
    int* dim_o = out->dims;
    char* name = out->name;
    CLTensor* otensor = new CLTensor();
    int TengineDataLayOut = out->layout;

    if (bForcedNHWCMode_ == true && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    else if (bForcedNHWCMode_ == false && TengineDataLayOut == TENGINE_LAYOUT_NCHW)
    {
        // need to re init datalayout to nhwc
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[3], dim_o[2], dim_o[1], dim_o[0]), 1, data_type_);
        //        ClTensorInfo_o.set_data_layout(DataLayout::NHWC);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    else
    {
        TensorInfo ClTensorInfo_o = TensorInfo(TensorShape(dim_o[1], dim_o[3], dim_o[2], dim_o[0]), 1, data_type_);
        ClTensorInfo_o.set_data_layout(DataLayout::NCHW);
        otensor->allocator()->init(ClTensorInfo_o);
    }
    tensors_map_[name] = otensor;

    CLConcatenateLayer* concat = new CLConcatenateLayer();
    // concat->configure(inputs_vector, otensor, DataLayoutDimension::CHANNEL);
    int axis = 3 - param->axis;
    concat->configure(inputs_vector, otensor, axis);
    functions_map_.push_back(concat);

    return true;
}
