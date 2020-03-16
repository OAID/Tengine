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

extern void RegisterCopy_NodeExec(void);
extern void RegisterCrop_NodeExec(void);
extern void RegisterEluNodeExec(void);
extern void RegisterL2NormalizationNodeExec(void);
extern void RegisterL2Pool_NodeExec(void);
extern void RegisterLayerNormLSTMNodeExec(void);
extern void RegisterLogisticNodeExec(void);
extern void RegisterPower_NodeExec(void);
extern void RegisterScale_NodeExec(void);
extern void RegisterSliceNodeExec(void);
extern void RegisterStridedSliceNodeExec(void);
extern void RegisterUpsampleNodeExec(void);

void RegisterCommonOps(void)
{
    RegisterCopy_NodeExec();
    RegisterCrop_NodeExec();
    RegisterEluNodeExec();
    RegisterL2NormalizationNodeExec();
    RegisterL2Pool_NodeExec();
    RegisterLayerNormLSTMNodeExec();
    RegisterLogisticNodeExec();
    RegisterPower_NodeExec();
    RegisterScale_NodeExec();
    RegisterSliceNodeExec();
    RegisterStridedSliceNodeExec();
    RegisterUpsampleNodeExec();
}

}    // namespace TEngine
