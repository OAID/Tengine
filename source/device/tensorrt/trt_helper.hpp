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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>

#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                                  \
    do                                                                 \
    {                                                                  \
        auto ret = (status);                                           \
        if (ret != 0)                                                  \
        {                                                              \
            Log(Loglevel, "TensorRT Engine", "Cuda failure: %d", ret); \
            abort();                                                   \
        }                                                              \
    } while (0)

constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

class Logger : public nvinfer1::ILogger
{
public:
    nvinfer1::ILogger::Severity severity_;

public:
    Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : severity_(severity){};

    void log(Severity severity, const char* msg) override
    {
        if (severity <= this->severity_)
        {
            switch (severity)
            {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                fprintf(stderr, "Tengine Fatal: %s\n", msg);
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                fprintf(stderr, "Tengine Error: %s\n", msg);
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                fprintf(stderr, "Tengine Warning: %s\n", msg);
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                fprintf(stderr, "Tengine Info: %s\n", msg);
                break;
            default:
                fprintf(stderr, "Tengine Normal: %s\n", msg);
                break;
            }
        }
        else
        {
            return;
        }
    }

    nvinfer1::ILogger::Severity get_severity()
    {
        return this->severity_;
    }

    nvinfer1::ILogger& get_logger()
    {
        return *this;
    }
};

struct InferDeleter
{
    template<typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

inline void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            fprintf(stderr, "Tengine Warning: Trying to use DLA core on a platform that doesn't have any DLA cores.\n");
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
}

// Ensures that every tensor used by a network has a scale.
//
// All tensors in a network must have a range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales for the entire network.
//
// If a tensor does not have a scale, it is assigned inScales or outScales as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its scale is assigned inScales.
// * Otherwise its scale is assigned outScales.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where scaling factors are asymmetric. */*/
void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

struct CaffeBufferShutter
{
    ~CaffeBufferShutter()
    {
        nvcaffeparser1::shutdownProtobufLibrary();
    }
};

struct UffBufferShutter
{
    ~UffBufferShutter()
    {
        nvuffparser::shutdownProtobufLibrary();
    }
};

template<typename T>
using TensorRTSmartPoint = std::unique_ptr<T, InferDeleter>;

using TensorRTShapeRange = std::array<nvinfer1::Dims, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;
