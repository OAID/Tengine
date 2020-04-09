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

namespace TEngine {

extern void RegisterConv2dFast(void);
extern void RegisterConv2dDepth(void);
extern void RegisterConv2dDepth3x3(void);
extern void RegisterConv2dWinograd(void);
extern void RegisterConv2dWinograd_1(void);
extern void RegisterConv2dWinogradNHWC(void);
extern void RegisterFullyConnectedFast(void);
extern void RegisterPoolingNodeExec(void);
extern void RegisterPoolingFP32NHWCNodeExec(void);
extern void RegisterBatchNormNodeExec(void);
extern void RegisterScaleNodeExec(void);
extern void RegisterDeconvNodeExec(void);
extern void RegisterDeConv2dDepth(void);
extern void RegisterDeConv2dDepth4x4(void);
extern void RegisterLRNNodeExec(void);
extern void RegisterFeatureMatchFast(void);
extern void RegisterConv2d_DW_FP32_NHWC(void);
extern void RegisterConv2dFp32NHWC(void);
extern void RegisterSigmoidFP32(void);
extern void RegisterTanhFP32(void);
extern void RegisterConv2dDepthDilation(void);
extern void RegisterConv2dDepthK5(void);
extern void RegisterConv2dDepthK7(void);
extern void RegisterAbsvalFP32(void);
extern void RegisterPReluFP32(void);
extern void RegisterReluFP32(void);
extern void RegisterSoftmaxFP32(void);
extern void RegisterHardswishFP32(void);
extern void RegisterInterpFP32(void);
extern void RegisterSeluFP32(void);
extern void RegisterEluFP32(void);
extern void RegisterCastFP32(void);
extern void RegisterEltwiseFP32(void);
extern void RegisterConv2dDirect3x3Dilation(void);
extern void RegisterLRNNodeExec(void);

void __attribute__((visibility("default"))) RegisterArmOps(void)
{
    RegisterConv2dFast();
    RegisterConv2dDepth();
    RegisterConv2dDepth3x3();
    RegisterConv2dWinograd();
    RegisterConv2dWinograd_1();
    RegisterConv2dWinogradNHWC();
    RegisterFullyConnectedFast();
    RegisterFeatureMatchFast();
    RegisterPoolingNodeExec();
    RegisterPoolingFP32NHWCNodeExec();
    RegisterBatchNormNodeExec();
    RegisterScaleNodeExec();
    RegisterDeconvNodeExec();
    RegisterDeConv2dDepth();
    RegisterDeConv2dDepth4x4();
    RegisterLRNNodeExec();
    RegisterConv2d_DW_FP32_NHWC();
    RegisterConv2dFp32NHWC();
    RegisterSigmoidFP32();
    RegisterTanhFP32();
    RegisterConv2dDepthDilation();
    RegisterConv2dDepthK5();
    RegisterConv2dDepthK7();
    RegisterAbsvalFP32();
    RegisterPReluFP32();
    RegisterReluFP32();
    RegisterSoftmaxFP32();
    RegisterHardswishFP32();
    RegisterInterpFP32();
    RegisterSeluFP32();
    RegisterEluFP32();
    RegisterCastFP32();
    RegisterEltwiseFP32();
    RegisterConv2dDirect3x3Dilation();
	RegisterLRNNodeExec();
}

}    // namespace TEngine
