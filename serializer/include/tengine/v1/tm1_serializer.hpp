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
 * Author: jingyou@openailab.com
 */
#ifndef __TM1_SERIALIZER_HPP__
#define __TM1_SERIALIZER_HPP__

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

#include "tm1_format.h"
#include "tm_serializer.hpp"

namespace TEngine {

namespace TMSerializer1 {

class TmSerializer1 : public TmSerializer
{
    using name_map_t = std::unordered_map<std::string, unsigned int>;

public:
    TmSerializer1()
    {
        name_ = "tm_loader";
        version_ = "1.0";
        format_name_ = "tengine";
    }

    virtual ~TmSerializer1(){};

    bool LoadModelFromMem(void* mmap_buf, StaticGraph* graph) override;
    bool SaveModelIntoMem(void* start_ptr, Graph* graph, uint32_t* tm_model_size) override;

    bool LoadBinaryFile(const char* tm_fname, int& fd, void*& buf, int& size);
    bool LoadNode(StaticGraph* graph, StaticNode* node, const TM_Node* tm_node, void* mmap_buf);
    bool LoadTensor(StaticGraph* graph, const TM_Tensor* tm_tensor, const TM_Buffer* tm_buf, void* mmap_buf);
    bool LoadGraph(StaticGraph* graph, const TM_Model* tm_model, void* mmap_buf);

    tm_uoffset_t SaveTmSubgraph(void* const start_ptr, tm_uoffset_t* cur_pos, Graph* graph);
    tm_uoffset_t SaveTmNode(void* const start_ptr, tm_uoffset_t* cur_pos, Node* node, name_map_t& tensor_name_map);
    tm_uoffset_t SaveTmTensor(void* const start_ptr, tm_uoffset_t* cur_pos, Tensor* tensor, unsigned int tensor_id,
                              unsigned int buffer_id);

    bool IsSaveString(void);
    bool IsSaveData(void);
};

}    // namespace TMSerializer1

}    // namespace TEngine

#endif
