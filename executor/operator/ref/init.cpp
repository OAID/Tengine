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
#include <iostream>
#include <functional>

namespace TEngine {

extern void RegisterRefPoolingOps(void);
extern void RegisterRefConv2d(void);
extern void RegisterRefDeconv2d(void);
extern void RegisterRefSoftmaxOps(void);
extern void RegisterRefDetectionPostOps(void);
extern void RegisterRefFCOps(void);
extern void RegisterRelu6Ops(void);
extern void RegisterReluOps(void);
extern void RegisterPreluOps(void);
extern void RegisterTanhOps(void);
extern void RegisterSigmoidOps(void);
extern void RegisterResizeOps(void);
extern void RegisterFlattenOps(void);
extern void RegisterReshapeOps(void);
extern void RegisterDropoutOps(void);
extern void RegisterRefConcat(void);
extern void RegisterRefPermute(void);
extern void RegisterRefLrn(void);
extern void RegisterEltwiseOps(void);
extern void RegisterRefSlice(void);
extern void RegisterSplitOps(void);
extern void RegisterPadOps(void);
extern void RegisterReductionOps(void);
extern void RegisterSqueezeOps(void);
extern void RegisterSwapAxisOps(void);
extern void RegisterRefRPNOps(void);
extern void RegisterRefBatchNormOps(void);
extern void RegisterRefNormlizeOps(void);
extern void RegisterRefAddNOps(void);

void RegisterRefOps(void)
{
    RegisterRefPoolingOps();
    RegisterRefConv2d();
    RegisterRefDeconv2d();
    RegisterRefSoftmaxOps();
    RegisterRefDetectionPostOps();
    RegisterRefFCOps();
    RegisterRefConcat();
    RegisterRefPermute();
    RegisterRelu6Ops();
    RegisterReluOps();
    RegisterPreluOps();
    RegisterTanhOps();
    RegisterSigmoidOps();
    RegisterResizeOps();
    RegisterFlattenOps();
    RegisterReshapeOps();
    RegisterDropoutOps();
    RegisterRefLrn();
    RegisterEltwiseOps();
    RegisterRefSlice();
    RegisterSplitOps();
    RegisterPadOps();
    RegisterReductionOps();
    RegisterSqueezeOps();
    RegisterSwapAxisOps();
    RegisterRefRPNOps();
    RegisterRefBatchNormOps();
    RegisterRefNormlizeOps();
    RegisterRefAddNOps();

}

}    // namespace TEngine
