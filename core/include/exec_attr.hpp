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
 * Author: haitao@openailab.com
 */
#ifndef __EXEC_ATTR_HPP__
#define __EXEC_ATTR_HPP__

namespace TEngine {

enum exec_policy_t
{
    kExecLatency,
    kExecLowPower
};

/*
   How to decide if an INT8 kernel should be used?

   when kernel_mode is EXEC_KERNEL_INT8

   How to decide if the output format of INT8 kernel?
   should it be float16 or float32 or int32 or int8?

   just follow the the output_tensor data_type!
*/

#define EXEC_KERNEL_FP32 0
#define EXEC_KERNEL_FP16 1
#define EXEC_KERNEL_INT8 2
#define EXEC_KERNEL_UINT8 3

#define MODEL_FORMAT_UNKNOWN 0
#define MODEL_FORMAT_TENGINE 1
#define MODEL_FORMAT_CAFFE 2
#define MODEL_FORMAT_ONNX 3
#define MODEL_FORMAT_MXNET 4
#define MODEL_FORMAT_TENSORFLOW 5
#define MODEL_FORMAT_TFLITE 6
#define MODEL_FORMAT_DARKNET 7
#define MODEL_FORMAT_DLA 8
#define MODEL_FORMAT_NCNN 9


#define MODEL_SUBFORMAT_AIPU 1
#define MODEL_SUBFORMAT_NNIE 2

struct ExecAttr
{
    exec_policy_t policy;
    int priority;
    int kernel_mode;
    int model_format;
    int model_layout;
    int graph_layout;
    bool low_mem_mode;
    bool fc_mt;    // fc should in multi-threaded?
    bool pooling_mt;    // pooling should in multi-threaded?
    void* exec_context;
    void* dev_handle;

    ExecAttr(void)
    {
        policy = kExecLatency;
        priority = 100;
        kernel_mode = EXEC_KERNEL_FP32;
        low_mem_mode = true;
        fc_mt = false;
        pooling_mt = true;
        model_format = MODEL_FORMAT_TENGINE;
        exec_context = nullptr;
        dev_handle = nullptr;
        graph_layout = -1;
        model_layout = -1;
    }
};

}    // namespace TEngine

#endif
