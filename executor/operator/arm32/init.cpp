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
extern void RegisterConv2dWinograd(void);
extern void RegisterConv2dWinogradNHWC(void);
extern void RegisterConv2dFp32NHWC(void);
extern void RegisterConv2dDepth3x3(void);
extern void RegisterConv2dDepth(void);
extern void RegisterFullyConnectedFast(void);
extern void RegisterDeconvNodeExec(void);
extern void RegisterDeConv2dDepth(void);
extern void RegisterDeConv2dDepth4x4(void);
extern void RegisterConv2d_DW_FP32_NHWC(void);
extern void RegisterConv2dDepthDilation(void);
extern void RegisterConv2dDepthK5(void);
extern void RegisterConv2dDepthK7(void);
extern void RegisterConv2dDirect3x3Dilation(void);

void __attribute__((visibility("default"))) RegisterArmOps(void)
{
    RegisterConv2dFast();
    RegisterConv2dWinograd();
    RegisterConv2dWinogradNHWC();
    RegisterConv2dFp32NHWC();
    RegisterConv2dDepth3x3();
    RegisterConv2dDepth();
    RegisterFullyConnectedFast();
    RegisterDeconvNodeExec();
    RegisterDeConv2dDepth();
    RegisterDeConv2dDepth4x4();
    RegisterConv2d_DW_FP32_NHWC();
    RegisterConv2dDepthDilation();
    RegisterConv2dDepthK5();
    RegisterConv2dDepthK7();
    RegisterConv2dDirect3x3Dilation();
}

}    // namespace TEngine
