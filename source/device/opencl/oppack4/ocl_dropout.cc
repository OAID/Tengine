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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hbshi@openailab.com
 */

#include "ocl_dropout.hpp"
ocl_dropout::ocl_dropout(OCLEngine* engine, struct node* ir_node)
    : ocl_node(engine, ir_node)
{
}
void ocl_dropout::pre_run()
{
    int ir_tensor_idx_output = ir_node->output_tensors[0];
    int ir_tensor_idx_input = ir_node->input_tensors[0];
    uint64_t handle_input = engine->get_gpu_mem_by_idx(ir_tensor_idx_input);
    engine->set_gpu_mem_by_idx(ir_tensor_idx_output, handle_input);
}
void ocl_dropout::run(struct subgraph* subgraph)
{

}

class ocl_dropout_creator : public ocl_node_creator
{
public:
    ocl_node* creator(OCLEngine* engine, struct node* ir_node) override
    {
        return new ocl_dropout(engine, ir_node);
    }
};

REGISTER_OCL_OP(OP_DROPOUT, ocl_dropout_creator);
