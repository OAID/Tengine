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
#ifndef __CAFFE_SERIALIZER_HPP__
#define __CAFFE_SERIALIZER_HPP__

#include <iostream>
#include <fstream>
#include <functional>

#include "te_caffe.pb.h"

#include "serializer.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

namespace TEngine {

class CaffeSingle : public Serializer
{
public:
    using name_map_t = std::unordered_map<std::string, std::string>;

    CaffeSingle()
    {
        name_ = "caffe_loader";
        version_ = "0.1";
        format_name_ = "caffe";
    }

    virtual ~CaffeSingle() {}

    virtual unsigned int GetFileNum(void) override
    {
        return 1;
    }

    virtual bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

protected:
    bool LoadBinaryFile(const char* fname, te_caffe::NetParameter& caffe_net);
    bool LoadTextFile(const char* fname, te_caffe::NetParameter& caffe_net);

    virtual bool LoadGraph(te_caffe::NetParameter& model, StaticGraph* graph);
    bool LoadNode(StaticGraph* graph, StaticNode*, const te_caffe::LayerParameter&, name_map_t&);
};

class CaffeBuddy : public CaffeSingle
{
public:
    CaffeBuddy()
    {
        name_ = "caffe_buddy";
    }

    unsigned int GetFileNum(void) override
    {
        return 2;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* static_graph) override;
    bool LoadModel(const std::vector<const void*>& addr_list, const std::vector<int>& size_list,
                   StaticGraph* static_graph, bool transfer_mem) override;

    using CaffeSingle::LoadGraph;
protected:
    bool LoadGraph(te_caffe::NetParameter& test_net, te_caffe::NetParameter& train_net, StaticGraph* graph);
};

}    // namespace TEngine

#endif
