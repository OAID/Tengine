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
 * Author: jingyou@openailab.com
 */
#ifndef __TM_SERIALIZER_HPP__
#define __TM_SERIALIZER_HPP__

#include "serializer.hpp"
#include "static_graph_interface.hpp"

namespace TEngine {

class TmSerializer : public Serializer
{
public:
    TmSerializer(){};
    virtual ~TmSerializer(){};

    unsigned int GetFileNum(void) override
    {
        return 1;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;
    bool SaveModel(const std::vector<std::string>& file_list, Graph* graph) override;
    bool LoadModel(const std::vector<const void*>& addr_list, const std::vector<int>& size_list,
                   StaticGraph* static_graph, bool transfer_mem) override;
    bool SaveModel(std::vector<void*>& addr_list, std::vector<int>& size_list, Graph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

    bool LoadBinaryFile(const char* tm_fname, int& fd, void*& buf, int& size);

    virtual bool LoadModelFromMem(void* mmap_buf, StaticGraph* graph)
    {
        return false;
    }
    virtual bool SaveModelIntoMem(void* start_ptr, Graph* graph, uint32_t* tm_model_size)
    {
        return false;
    }
};

using TmSerializerPtr = std::shared_ptr<TmSerializer>;
using TmSerializerFactory = SpecificFactory<TmSerializer>;

extern template class SpecificFactory<TmSerializer>;
extern template SpecificFactory<TmSerializer> SpecificFactory<TmSerializer>::instance;

class TmSerializerManager : public SimpleObjectManagerWithLock<TmSerializerManager, TmSerializerPtr>
{
};

}    // namespace TEngine

#endif
