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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __SERIALIZER_HPP__
#define __SERIALIZER_HPP__

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>

#include "static_graph_interface.hpp"
#include "generic_factory.hpp"
#include "safe_object_manager.hpp"
#include "graph.hpp"

namespace TEngine {

class Serializer
{
public:
    using op_load_map_t = std::unordered_map<std::string, any>;
    using op_save_map_t = std::unordered_map<std::string, any>;

    Serializer() {}
    virtual ~Serializer(){};

    virtual unsigned int GetFileNum(void) = 0;
    virtual bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph) = 0;
    virtual bool SaveModel(const std::vector<std::string>& file_list, Graph* graph)
    {
        return false;
    }

    /* the memory stored in addr_list will be released by static graph */
    virtual bool LoadModel(const std::vector<const void*>& addr_list, const std::vector<int>& size_list,
                           StaticGraph* static_graph, bool transfer_mem = false)
    {
        return false;
    }
    virtual bool SaveModel(std::vector<void*>& addr_list, std::vector<int>& size_list, Graph* graph)
    {
        return false;
    }

    /* interface to load const tensor */
    virtual bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) = 0;
    virtual bool LoadConstTensor(int fd, StaticTensor* const_tensor) = 0;

    const std::string& GetFormatName(void)
    {
        return format_name_;
    }
    const std::string& GetName(void)
    {
        return name_;
    }
    const std::string& GetVersion(void)
    {
        return version_;
    }

    /* helper functions */
    bool RegisterOpLoadMethod(const std::string& op_name, const any& load_func)
    {
        if(op_load_map_.count(op_name))
            return false;

        op_load_map_[op_name] = load_func;
        return true;
    }

    bool FindOpLoadMethod(const std::string& op_name)
    {
        if(op_load_map_.count(op_name))
            return true;

        return false;
    }

    any& GetOpLoadMethod(const std::string& op_name)
    {
        return op_load_map_[op_name];
    }

    bool RegisterOpSaveMethod(const std::string& op_name, const any& save_func)
    {
        if(op_save_map_.count(op_name))
            return false;

        op_save_map_[op_name] = save_func;
        return true;
    }

    bool FindOpSaveMethod(const std::string& op_name)
    {
        if(op_save_map_.count(op_name))
            return true;

        return false;
    }

    any& GetOpSaveMethod(const std::string& op_name)
    {
        return op_save_map_[op_name];
    }

protected:
    std::string version_;
    std::string name_;
    std::string format_name_;
    op_load_map_t op_load_map_;
    op_save_map_t op_save_map_;
};

using SerializerPtr = std::shared_ptr<Serializer>;
using SerializerFactory = SpecificFactory<Serializer>;

extern template class SpecificFactory<Serializer>;
extern template SpecificFactory<Serializer> SpecificFactory<Serializer>::instance;

class SerializerManager;
extern template SerializerManager SimpleObjectManagerWithLock<SerializerManager, SerializerPtr>::instance;

class SerializerManager : public SimpleObjectManagerWithLock<SerializerManager, SerializerPtr>
{
};

}    // namespace TEngine

#endif
