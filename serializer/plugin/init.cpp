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

<<<<<<< HEAD
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

#ifdef CONFIG_TFLITE_SERIALIZER
#include "tf_lite_serializer.hpp"
#endif

#ifdef CONFIG_TENGINE_SERIALIZER
#include "tm_serializer.hpp"
#include "src_tm_serializer.hpp"
=======
#ifdef CONFIG_TENGINE_SERIALIZER
#include "tm_serializer.hpp"
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
#endif

#include "logger.hpp"

namespace TEngine {

<<<<<<< HEAD
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

#ifdef CONFIG_TFLITE_SERIALIZER
extern bool TFLiteSerializerRegisterOpLoader();
#endif

=======
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
#ifdef CONFIG_TENGINE_SERIALIZER
bool TmSerializerInit(void);
#endif

}    // namespace TEngine

using namespace TEngine;

int serializer_plugin_init(void)
{
    // Register into factory

<<<<<<< HEAD
    auto factory = SerializerFactory::GetFactory();

#ifdef CONFIG_ONNX_SERIALIZER
    factory->RegisterInterface<OnnxSerializer>("onnx");
    auto onnx_serializer = factory->Create("onnx");

    SerializerManager::SafeAdd("onnx", SerializerPtr(onnx_serializer));
    OnnxSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_CAFFE_SERIALIZER
    factory->RegisterInterface<CaffeSingle>("caffe_single");
    factory->RegisterInterface<CaffeBuddy>("caffe_buddy");

    auto caffe_single = factory->Create("caffe_single");
    auto caffe_buddy = factory->Create("caffe_buddy");

    SerializerManager::SafeAdd("caffe_single", SerializerPtr(caffe_single));
    SerializerManager::SafeAdd("caffe", SerializerPtr(caffe_buddy));

    CaffeSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_MXNET_SERIALIZER
    factory->RegisterInterface<MxnetSerializer>("mxnet");
    auto mxnet_serializer = factory->Create("mxnet");

    SerializerManager::SafeAdd("mxnet", SerializerPtr(mxnet_serializer));

    MxnetSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TF_SERIALIZER
    factory->RegisterInterface<TFSerializer>("tensorflow");
    auto tf_serializer = factory->Create("tensorflow");

    SerializerManager::SafeAdd("tensorflow", SerializerPtr(tf_serializer));

    TFSerializerRegisterOpLoader();
#endif

#ifdef CONFIG_TFLITE_SERIALIZER
    factory->RegisterInterface<TFLiteSerializer>("tflite");
    auto tf_lite_serializer = factory->Create("tflite");

    SerializerManager::SafeAdd("tflite", SerializerPtr(tf_lite_serializer));

    TFLiteSerializerRegisterOpLoader();
#endif
=======
    //auto factory = SerializerFactory::GetFactory();
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

#ifdef CONFIG_TENGINE_SERIALIZER
    TmSerializerInit();

<<<<<<< HEAD
#define SrcTmName "src_tm"
=======
/*#define SrcTmName "src_tm"
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

    factory->RegisterInterface<SrcTmSerializer>(SrcTmName);
    auto src_tm_serializer = factory->Create(SrcTmName);

<<<<<<< HEAD
    SerializerManager::SafeAdd(SrcTmName, SerializerPtr(src_tm_serializer));
=======
    SerializerManager::SafeAdd(SrcTmName, SerializerPtr(src_tm_serializer));*/
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

#endif

    // std::cout<<"SERIALIZER PLUGIN INITED\n";

    return 0;
}
