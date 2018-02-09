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
#include <iostream>
#include <functional>

#include "caffe_serializer.hpp"
#include "onnx_serializer.hpp"
#include "mxnet_serializer.hpp"
#include "logger.hpp"

namespace TEngine {

extern bool OnnxSerializerRegisterOpLoader();
extern bool CaffeSerializerRegisterOpLoader();
extern bool MxnetSerializerRegisterOpLoader();

}

extern "C" {

   int tengine_plugin_init(void);
}

using namespace TEngine;

int tengine_plugin_init(void)
{
    //Register into factory

    auto factory=SerializerFactory::GetFactory();

    factory->RegisterInterface<OnnxSerializer>("onnx");
    factory->RegisterInterface<CaffeSingle>("caffe_single");
    factory->RegisterInterface<CaffeBuddy>("caffe_buddy");
    factory->RegisterInterface<MxnetSerializer>("mxnet");

    //create the serializer object by factory...
    auto onnx_serializer=factory->Create("onnx");
    auto caffe_single=factory->Create("caffe_single");
    auto caffe_buddy=factory->Create("caffe_buddy");
    auto mxnet_serializer=factory->Create("mxnet");

    //add into object store
    SerializerManager::SafeAdd("onnx",SerializerPtr(onnx_serializer));
    SerializerManager::SafeAdd("caffe_single",SerializerPtr(caffe_single));
    SerializerManager::SafeAdd("caffe",SerializerPtr(caffe_buddy));
    SerializerManager::SafeAdd("mxnet",SerializerPtr(mxnet_serializer));

    //add the operator loader
    OnnxSerializerRegisterOpLoader();
    CaffeSerializerRegisterOpLoader();
    MxnetSerializerRegisterOpLoader();

    std::cout<<"SERIALIZER PLUGIN INITED\n";   

    return 0;
}

