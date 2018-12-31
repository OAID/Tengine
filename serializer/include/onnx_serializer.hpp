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
#ifndef __ONNX_SERIALIER_HPP__
#define __ONNX_SERIALIER_HPP__

#include <iostream>
#include <fstream>
#include <functional>
#include <unordered_map>

#include "serializer.hpp"
#include "static_graph_interface.hpp"

#include "logger.hpp"
#include "onnx.pb.h"

namespace TEngine {

class OnnxSerializer : public Serializer
{
public:
    OnnxSerializer()
    {
        name_ = "onnx_loader";
        format_name_ = "onnx";
        version_ = "0.1";
    }

    virtual ~OnnxSerializer(){};

    unsigned int GetFileNum(void) override
    {
        return 1;
    }

    bool LoadModel(const std::vector<std::string>& file_list, StaticGraph* graph) override;

    bool LoadConstTensor(const std::string& fname, StaticTensor* const_tensor) override
    {
        return false;
    }
    bool LoadConstTensor(int fd, StaticTensor* const_tensor) override
    {
        return false;
    }

protected:
    bool LoadModelFile(const char* fname, onnx::ModelProto& model);

    bool LoadGraph(onnx::ModelProto& model, StaticGraph* graph);
    bool LoadConstTensor(StaticGraph* graph, const onnx::GraphProto& onnx_graph);
    void CreateInputNode(StaticGraph* graph, const onnx::GraphProto& onnx_graph);
    bool LoadNode(StaticGraph* graph, StaticNode*, const onnx::NodeProto&);
};

}    // namespace TEngine

#endif
