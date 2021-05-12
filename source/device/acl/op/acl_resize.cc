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
#include "operator/op.h"
}

bool CLGraph::AddResizeLayer(struct node* node)
{
#ifdef ACL_EXTENSTION
    struct graph* graph = node->graph;
        struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
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
        struct tensor* o_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
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
