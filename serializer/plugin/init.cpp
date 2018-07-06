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

#include "serializer.hpp"

#ifdef CONFIG_CAFFE_SERIALIZER
#include "caffe_serializer.hpp"
#endif

#ifdef CONFIG_ONNX_SERIALIZER
#include "onnx_serializer.hpp"
#endif

#ifdef CONFIG_MXNET_SERIALIZER
#include "mxnet_serializer.hpp"
#endif

#ifdef CONFIG_TF_SERIALIZER
#include "tf_serializer.hpp"
#endif

#ifdef CONFIG_TENGINE_SERIALIZER
#include "tm_serializer.hpp"
#endif

#include "logger.hpp"

namespace TEngine {

#ifdef CONFIG_ONNX_SERIALIZER
extern bool OnnxSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_CAFFE_SERIALIZER
extern bool CaffeSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_MXNET_SERIALIZER
extern bool MxnetSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TF_SERIALIZER
extern bool TFSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TENGINE_SERIALIZER
extern bool TmSerializerRegisterOpLoader();
#endif

}


extern "C" {

   int serializer_plugin_init(void);
}

using namespace TEngine;

int serializer_plugin_init(void)
{
    //Register into factory

    auto factory=SerializerFactory::GetFactory();

#ifdef CONFIG_ONNX_SERIALIZER
    factory->RegisterInterface<OnnxSerializer>("onnx");
    auto onnx_serializer=factory->Create("onnx");

    SerializerManager::SafeAdd("onnx",SerializerPtr(onnx_serializer));
    OnnxSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_CAFFE_SERIALIZER
    factory->RegisterInterface<CaffeSingle>("caffe_single");
    factory->RegisterInterface<CaffeBuddy>("caffe_buddy");

    auto caffe_single=factory->Create("caffe_single");
    auto caffe_buddy=factory->Create("caffe_buddy");

    SerializerManager::SafeAdd("caffe_single",SerializerPtr(caffe_single));
    SerializerManager::SafeAdd("caffe",SerializerPtr(caffe_buddy));

    CaffeSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_MXNET_SERIALIZER
    factory->RegisterInterface<MxnetSerializer>("mxnet");
    auto mxnet_serializer=factory->Create("mxnet");

    SerializerManager::SafeAdd("mxnet",SerializerPtr(mxnet_serializer));

    MxnetSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TF_SERIALIZER
    factory->RegisterInterface<TFSerializer>("tensorflow");
    auto tf_serializer=factory->Create("tensorflow");

    SerializerManager::SafeAdd("tensorflow",SerializerPtr(tf_serializer));

    TFSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TENGINE_SERIALIZER
    factory->RegisterInterface<TmSerializer>("tengine");
    auto tm_serializer=factory->Create("tengine");

    SerializerManager::SafeAdd("tengine",SerializerPtr(tm_serializer));

    TmSerializerRegisterOpLoader();
#endif

    //std::cout<<"SERIALIZER PLUGIN INITED\n";   

    return 0;
}

