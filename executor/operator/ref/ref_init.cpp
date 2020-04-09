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
extern void RegisterRefReorg(void);
extern void RegisterRefRegion(void);
extern void RegisterRefRoiPooling(void);
extern void RegisterRefPriorBox(void);
extern void RegisterRefClipOps(void);
extern void RegisterRefTileOps(void);
extern void RegisterRefDetectionOutput(void);
extern void RegisterRefMaximumOps(void);
extern void RegisterRefMinimumOps(void);
extern void RegisterRefArgMaxOps(void);
extern void RegisterRefArgMinOps(void);
extern void RegisterTopkV2Ops(void);
extern void RegisterRefShuffleChannel(void);
extern void RegisterRefBatchToSpaceND(void);
extern void RegisterRefSpaceToBatchND(void);
extern void RegisterRefLogSoftmaxOps(void);
extern void RegisterRefExpandDimsOps(void);
extern void RegisterRefUnaryOps(void);
extern void RegisterRefRoialignOps(void);
extern void RegisterRefPsroipoolingOps(void);
extern void RegisterRefBiasOps(void);
extern void RegisterRefNoopOps(void);
extern void RegisterRefThresholdOps(void);
extern void RegisterRefHardsigmoidOps(void);
extern void RegisterRefEmbedOps(void);
extern void RegisterRefInstanceNormOps(void);
extern void RegisterRefMVNOps(void);
extern void RegisterRefBroadMulOps(void);
extern void RegisterRefLogicalOps(void);
extern void RegisterRefGatherOps(void);
extern void RegisterRefTransposeOps(void);
extern void RegisterRefReverseOps(void);
extern void RegisterComparisonOps(void);
extern void RegisterRefSpaceToDepth(void);
extern void RegisterRefDepthToSpace(void);
extern void RegisterRefSquaredDifferenceOps(void);
extern void RegisterRefSparseToDenseOps(void);
extern void RegisterRefCeilOps(void);
extern void RegisterRefRoundOps(void);
extern void RegisterRefZerosLikeOps(void);
extern void RegisterRefInterpOps(void);
extern void RegisterRefLogisticOps(void);
extern void RegisterRefFeatureMatchOps(void);
extern void RegisterRefL2NormOps(void);
extern void RegisterRefL2PoolOps(void);
extern void RegisterRefEluOps(void);
extern void RegisterRefCopyOps(void);
extern void RegisterRefLayernormLSTMOps(void);
extern void RegisterRefCropOps(void);
extern void RegisterRefPowerOps(void);
extern void RegisterRelu1Ops(void);
extern void RegisterRefScale(void);
extern void RegisterRefStridedSlice(void);
extern void RegisterRefUpsample(void);
extern void RegisterRefGRUOps(void);
extern void RegisterRefLSTMOps(void);
extern void RegisterRefRNNOps(void);
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
    RegisterRefReorg();
    RegisterRefRegion();
    RegisterRefRoiPooling();
    RegisterRefPriorBox();
    RegisterRefClipOps();
    RegisterRefTileOps();
    RegisterRefDetectionOutput();
    RegisterRefMaximumOps();
    RegisterRefMinimumOps();
    RegisterRefArgMaxOps();
    RegisterRefArgMinOps();
    RegisterTopkV2Ops();
    RegisterRefShuffleChannel();
    RegisterRefSpaceToBatchND(); 
    RegisterRefBatchToSpaceND();    
    RegisterRefLogSoftmaxOps();
    RegisterRefExpandDimsOps();
    RegisterRefUnaryOps();
    RegisterRefPsroipoolingOps();
    RegisterRefRoialignOps();
    RegisterRefBiasOps();
    RegisterRefNoopOps();
    RegisterRefThresholdOps();
    RegisterRefHardsigmoidOps();
    RegisterRefEmbedOps();
    RegisterRefInstanceNormOps();
    RegisterRefMVNOps();   
    RegisterRefBroadMulOps(); 
    RegisterRefLogicalOps();
    RegisterRefGatherOps();
    RegisterRefTransposeOps(); 
    RegisterRefReverseOps();    
    RegisterComparisonOps();       
    RegisterRefSpaceToDepth();
    RegisterRefDepthToSpace();
    RegisterRefSparseToDenseOps();
    RegisterRefCeilOps();
    RegisterRefRoundOps();
    RegisterRefSquaredDifferenceOps();
    RegisterRefZerosLikeOps();
    RegisterRefInterpOps();

    RegisterRefLogisticOps();
    RegisterRefFeatureMatchOps();
    RegisterRefL2NormOps();
    RegisterRefL2PoolOps();
    RegisterRefEluOps();
    RegisterRefCopyOps();
    RegisterRefLayernormLSTMOps();
    RegisterRefCropOps();
    RegisterRefPowerOps();
    RegisterRelu1Ops();
    RegisterRefScale();
    RegisterRefStridedSlice();
    RegisterRefUpsample();
    RegisterRefGRUOps();
    RegisterRefLSTMOps();
    RegisterRefRNNOps();
}

}    // namespace TEngine
