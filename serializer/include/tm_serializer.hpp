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
#ifndef __TM_SERIALIZER_HPP__
#define __TM_SERIALIZER_HPP__

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"
#include "tm_generate.h"

namespace TEngine {

class TmSerializer : public Serializer
{
    using name_map_t = std::unordered_map<std::string, unsigned int>;

public:
    TmSerializer()
    {
        name_ = "tm_loader";
        version_ = "0.1";
        format_name_ = "tengine";
    }

    virtual ~TmSerializer(){};

    unsigned int GetFileNum(void) override
    {
        return 1;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;
    bool SaveModel(const std::vector<std::string>& file_list, Graph* graph) override;
    bool LoadModel(const std::vector<const void*>& addr_list, const std::vector<int>& size_list,
                   StaticGraph* static_graph) override;
    bool SaveModel(std::vector<void*>& addr_list, std::vector<int>& size_list, Graph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

    bool LoadModelFromMem(void* mmap_buf, StaticGraph* graph);

    bool IsSaveString(void);
    bool IsSaveData(void);

protected:
    bool LoadBinaryFile(const char* tm_fname, int& fd, void*& buf, int& size);
    bool LoadNode(StaticGraph* graph, StaticNode* node, const TM_Node* tm_node, void* mmap_buf);
    bool LoadTensor(StaticGraph* graph, const TM_Tensor* tm_tensor, const TM_Buffer* tm_buf, void* mmap_buf);
    bool LoadGraph(StaticGraph* graph, const TM_Model* tm_model, void* mmap_buf);

    tm_uoffset_t SaveTmSubgraph(void* const start_ptr, tm_uoffset_t* cur_pos, Graph* graph);
    tm_uoffset_t SaveTmNode(void* const start_ptr, tm_uoffset_t* cur_pos, Node* node, name_map_t& tensor_name_map);
    tm_uoffset_t SaveTmTensor(void* const start_ptr, tm_uoffset_t* cur_pos, Tensor* tensor, unsigned int tensor_id,
                              unsigned int buffer_id);
};

}    // namespace TEngine

#endif
